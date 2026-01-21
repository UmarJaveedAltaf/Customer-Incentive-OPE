# analysis_ope_plots.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.config import TrainConfig
from src.agents.dqn_agent import DQNAgent
from src.ope.models import build_default_reward_model, flatten_episode_data

RULE_BASELINE_MEAN = 700.0
RISK_MARGIN = 0.0

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

def greedy_target_probs(agent, states: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    T = len(states)
    pi = np.zeros((T, agent.action_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch = states[i:i + batch_size]
            st = torch.tensor(batch, dtype=torch.float32, device=agent.device)
            q = agent.policy_net(st)
            greedy = q.argmax(dim=1).cpu().numpy()
            pi[np.arange(i, i + len(batch)), greedy] = 1.0
    return pi

def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 800, seed: int = 0, desc: str = "Bootstrapping") -> dict:
    vals = np.asarray(values, dtype=np.float32)
    n = len(vals)
    rng = np.random.default_rng(seed)
    estimate = float(vals.mean()) if n else float("nan")
    boots = np.empty(n_boot, dtype=np.float32)
    for b in tqdm(range(n_boot), desc=desc):
        idx = rng.integers(0, n, size=n)
        boots[b] = vals[idx].mean()
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": n,
        "n_boot": n_boot,
    }

def _ips_episode(rewards, pi_taken, b_taken, clip_rho=50.0):
    b_taken = np.clip(b_taken, 1e-8, 1.0)
    rho = np.clip(pi_taken / b_taken, 0.0, clip_rho)
    return float(np.sum(rho * rewards))

def _wis_episode(rewards, pi_taken, b_taken, clip_rho=50.0):
    b_taken = np.clip(b_taken, 1e-8, 1.0)
    rho = np.clip(pi_taken / b_taken, 0.0, clip_rho)
    w = rho / (np.sum(rho) + 1e-8)
    return float(np.sum(w * rewards))

def _dm_episode(pi_all, rhat_all):
    return float(np.sum(np.sum(pi_all * rhat_all, axis=1)))

def _dr_episode(actions, rewards, pi_all, pi_taken, b_taken, qhat_all, clip_rho=50.0):
    b_taken = np.clip(b_taken, 1e-8, 1.0)
    rho = np.clip(pi_taken / b_taken, 0.0, clip_rho)
    exp_q = np.sum(pi_all * qhat_all, axis=1)
    q_taken = qhat_all[np.arange(len(actions)), actions]
    return float(np.sum(exp_q + rho * (rewards - q_taken)))

def plot_ci_bars(results: dict, save_path: str):
    names = list(results.keys())
    means = [results[k]["estimate"] for k in names]
    lows = [results[k]["ci_low"] for k in names]
    highs = [results[k]["ci_high"] for k in names]

    yerr_low = [m - lo for m, lo in zip(means, lows)]
    yerr_high = [hi - m for m, hi in zip(means, highs)]

    plt.figure()
    x = np.arange(len(names))
    plt.bar(x, means, yerr=[yerr_low, yerr_high], capsize=6)
    plt.axhline(RULE_BASELINE_MEAN + RISK_MARGIN, linestyle="--")
    plt.xticks(x, names)
    plt.ylabel("Estimated return")
    plt.title("OPE Mean + 95% CI (Bootstrap)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_distributions(ips, wis, dm, dr, save_path: str):
    plt.figure()
    plt.hist(ips, alpha=0.5, bins=30, label="IPS")
    plt.hist(wis, alpha=0.5, bins=30, label="WIS")
    plt.hist(dm, alpha=0.5, bins=30, label="DM")
    plt.hist(dr, alpha=0.5, bins=30, label="DR")
    plt.axvline(RULE_BASELINE_MEAN + RISK_MARGIN, linestyle="--", label="Gate baseline")
    plt.legend()
    plt.xlabel("Per-episode estimate")
    plt.ylabel("Count")
    plt.title("Per-episode OPE estimate distributions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    log_path = "data/logged_behavior.npz"
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Missing {log_path}. Run collect_logged_data first.")

    os.makedirs("figures", exist_ok=True)

    data = np.load(log_path, allow_pickle=True)
    episodes = {k: list(data[k]) for k in data.files}

    first_states = _to_float32_array(episodes["states"][0])
    first_b_all = _to_float32_array(episodes["b_probs_all"][0])
    state_dim = first_states.shape[1]
    action_dim = first_b_all.shape[1]

    cfg = TrainConfig()
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)
    agent.load("dqn_policy.pth")
    agent.policy_net.eval()

    # Step 1: normalize + pi
    states_list, actions_list, rewards_list, b_taken_list, pi_taken_list, pi_all_list = [], [], [], [], [], []
    print("Step 1/4: Computing target probs...")
    for states, actions, rewards, b_taken in tqdm(
        zip(episodes["states"], episodes["actions"], episodes["rewards"], episodes["b_probs"]),
        total=len(episodes["states"]),
        desc="Episodes"
    ):
        s = _to_float32_array(states)
        a = _to_int64_array(actions).reshape(-1)
        r = _to_float32_array(rewards).reshape(-1)
        b = _to_float32_array(b_taken).reshape(-1)

        pi_all = greedy_target_probs(agent, s)
        pi_taken = pi_all[np.arange(len(a)), a]

        states_list.append(s)
        actions_list.append(a)
        rewards_list.append(r)
        b_taken_list.append(b)
        pi_taken_list.append(pi_taken)
        pi_all_list.append(pi_all)

    # Step 2: IPS/WIS
    print("Step 2/4: IPS/WIS...")
    ips_vals, wis_vals = [], []
    for i in tqdm(range(len(states_list)), desc="IPS/WIS"):
        ips_vals.append(_ips_episode(rewards_list[i], pi_taken_list[i], b_taken_list[i]))
        wis_vals.append(_wis_episode(rewards_list[i], pi_taken_list[i], b_taken_list[i]))

    # Step 3: reward model
    print("Step 3/4: Reward model...")
    episodes_norm = {
        "states": states_list,
        "actions": actions_list,
        "rewards": rewards_list,
    }
    flat_states, flat_actions, flat_rewards = flatten_episode_data(episodes_norm)
    rmodel = build_default_reward_model(action_dim=action_dim, seed=0)
    rmodel.fit(flat_states, flat_actions, flat_rewards)

    # Step 4: DM/DR
    print("Step 4/4: DM/DR...")
    dm_vals, dr_vals = [], []
    for i in tqdm(range(len(states_list)), desc="DM/DR"):
        s = states_list[i]
        a = actions_list[i]
        r = rewards_list[i]
        pi_all = pi_all_list[i]

        qhat_all = np.column_stack([
            rmodel.predict(s, np.full((len(s),), act, dtype=np.int64))
            for act in range(action_dim)
        ]).astype(np.float32)

        dm_vals.append(_dm_episode(pi_all, qhat_all))
        dr_vals.append(_dr_episode(a, r, pi_all, pi_taken_list[i], b_taken_list[i], qhat_all))

    # Bootstrap summaries
    print("\nBootstrapping summaries...")
    res = {
        "IPS": bootstrap_mean_ci(np.array(ips_vals), seed=0, desc="Bootstrap IPS"),
        "WIS": bootstrap_mean_ci(np.array(wis_vals), seed=1, desc="Bootstrap WIS"),
        "DM":  bootstrap_mean_ci(np.array(dm_vals),  seed=2, desc="Bootstrap DM"),
        "DR":  bootstrap_mean_ci(np.array(dr_vals),  seed=3, desc="Bootstrap DR"),
    }

    print("\n=== OPE Summary ===")
    for k in ["IPS", "WIS", "DM", "DR"]:
        print(f"{k:>3}: {res[k]['estimate']:8.2f}  CI [{res[k]['ci_low']:8.2f}, {res[k]['ci_high']:8.2f}]")

    # Plots
    plot_ci_bars(res, "figures/ope_ci_bars.png")
    plot_distributions(np.array(ips_vals), np.array(wis_vals), np.array(dm_vals), np.array(dr_vals),
                       "figures/ope_distributions.png")

    print("\nSaved plots:")
    print(" - figures/ope_ci_bars.png")
    print(" - figures/ope_distributions.png")

if __name__ == "__main__":
    main()
