# offline_cql_train.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.config import TrainConfig
from src.models.dqn_network import DuelingDQN  # using your dueling net


def _to_float32_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(arr, axis=0)
        except Exception:
            arr = np.asarray(list(arr))
    return arr.astype(np.float32)


def _to_int64_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(arr, axis=0)
        except Exception:
            arr = np.asarray(list(arr))
    return arr.astype(np.int64)


def load_offline_dataset(path="data/logged_behavior.npz"):
    data = np.load(path, allow_pickle=True)
    episodes = {k: list(data[k]) for k in data.files}

    S, A, R, S2, D = [], [], [], [], []
    # If next_states/dones weren’t logged, we can reconstruct episode-next by shifting
    # We assume each episode is a trajectory list: states[t], actions[t], rewards[t]
    for states, actions, rewards in zip(episodes["states"], episodes["actions"], episodes["rewards"]):
        states = _to_float32_array(states)
        actions = _to_int64_array(actions).reshape(-1)
        rewards = _to_float32_array(rewards).reshape(-1)

        # next_states via shift; last next_state = zeros
        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]
        next_states[-1] = 0.0

        dones = np.zeros((len(states),), dtype=np.float32)
        dones[-1] = 1.0

        S.append(states)
        A.append(actions)
        R.append(rewards)
        S2.append(next_states)
        D.append(dones)

    S = np.concatenate(S, axis=0)
    A = np.concatenate(A, axis=0)
    R = np.concatenate(R, axis=0)
    S2 = np.concatenate(S2, axis=0)
    D = np.concatenate(D, axis=0)
    return S, A, R, S2, D


def minibatches(S, A, R, S2, D, batch_size, rng):
    n = len(S)
    idx = rng.permutation(n)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield S[j], A[j], R[j], S2[j], D[j]


def main():
    dataset_path = "data/logged_behavior.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}. Run collect_logged_data first.")

    os.makedirs("figures", exist_ok=True)

    S, A, R, S2, D = load_offline_dataset(dataset_path)
    state_dim = S.shape[1]
    action_dim = int(np.max(A)) + 1  # assumes actions are 0..K-1

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = DuelingDQN(state_dim, action_dim, cfg.hidden_dim).to(device)
    q_tgt = DuelingDQN(state_dim, action_dim, cfg.hidden_dim).to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=cfg.lr)

    # --- CQL hyperparams ---
    alpha = 1.0              # strength of conservative penalty (tune 0.1 → 5.0)
    n_epochs = 15            # offline epochs
    batch_size = cfg.batch_size
    gamma = cfg.gamma
    tau = 0.005              # soft target update
    rng = np.random.default_rng(0)

    losses, bellman_losses, cql_terms = [], [], []

    # Behavior distribution for later viz
    beh_counts = np.bincount(A, minlength=action_dim).astype(np.float32)
    beh_dist = beh_counts / (beh_counts.sum() + 1e-8)

    print(f"Offline dataset size: {len(S)} transitions")
    print("Training CQL...")

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        epoch_bell = 0.0
        epoch_cql = 0.0
        n_steps = 0

        for s, a, r, s2, d in tqdm(
            minibatches(S, A, R, S2, D, batch_size, rng),
            desc=f"Epoch {epoch}/{n_epochs}"
        ):
            s = torch.tensor(s, dtype=torch.float32, device=device)
            a = torch.tensor(a, dtype=torch.int64, device=device)
            r = torch.tensor(r, dtype=torch.float32, device=device)
            s2 = torch.tensor(s2, dtype=torch.float32, device=device)
            d = torch.tensor(d, dtype=torch.float32, device=device)

            # Q(s,a)
            q_all = q(s)                              # [B, A]
            q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

            # Target
            with torch.no_grad():
                q2_all = q_tgt(s2)
                q2_max = q2_all.max(dim=1).values
                y = r + (1.0 - d) * gamma * q2_max

            bellman = F.smooth_l1_loss(q_sa, y)

            # --- CQL conservative term ---
            # logsumexp over all actions - Q(s, a_taken)
            logsumexp = torch.logsumexp(q_all, dim=1)
            cql_penalty = (logsumexp - q_sa).mean()

            loss = bellman + alpha * cql_penalty

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step()

            # soft update target
            with torch.no_grad():
                for p, pt in zip(q.parameters(), q_tgt.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)

            epoch_loss += float(loss.item())
            epoch_bell += float(bellman.item())
            epoch_cql += float(cql_penalty.item())
            n_steps += 1

        losses.append(epoch_loss / max(n_steps, 1))
        bellman_losses.append(epoch_bell / max(n_steps, 1))
        cql_terms.append(epoch_cql / max(n_steps, 1))

        print(f"Epoch {epoch:02d} | loss={losses[-1]:.4f} bellman={bellman_losses[-1]:.4f} cql={cql_terms[-1]:.4f}")

    # Save model
    ckpt_path = "cql_policy.pth"
    torch.save(q.state_dict(), ckpt_path)
    print(f"\n✅ Saved offline CQL policy to {ckpt_path}")

    # -------------------------
    # Visualizations
    # -------------------------
    # 1) Loss curves
    plt.figure()
    plt.plot(losses, label="Total loss")
    plt.plot(bellman_losses, label="Bellman loss")
    plt.plot(cql_terms, label="CQL penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CQL Offline Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/cql_training_curves.png", dpi=200)
    plt.close()

    # 2) Action distribution (behavior vs learned greedy)
    with torch.no_grad():
        S_t = torch.tensor(S[:50000], dtype=torch.float32, device=device)  # cap for speed
        q_all = q(S_t)
        greedy = q_all.argmax(dim=1).cpu().numpy()

    pol_counts = np.bincount(greedy, minlength=action_dim).astype(np.float32)
    pol_dist = pol_counts / (pol_counts.sum() + 1e-8)

    plt.figure()
    x = np.arange(action_dim)
    plt.bar(x - 0.2, beh_dist, width=0.4, label="Behavior")
    plt.bar(x + 0.2, pol_dist, width=0.4, label="CQL policy")
    plt.xlabel("Action")
    plt.ylabel("Probability")
    plt.title("Action Distribution: Behavior vs Offline CQL Policy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/cql_action_dist.png", dpi=200)
    plt.close()

    print("\nSaved plots:")
    print(" - figures/cql_training_curves.png")
    print(" - figures/cql_action_dist.png")


if __name__ == "__main__":
    main()
