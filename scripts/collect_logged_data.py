# scripts/collect_logged_data.py
import os
import argparse
import numpy as np
import torch

from src.environment.user_retention_env import UserRetentionEnv
from src.utils.config import EnvConfig, TrainConfig  # adjust if your config class names differ
from src.agents.dqn_agent import DQNAgent


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def eps_greedy_probs(q_values: np.ndarray, eps: float) -> np.ndarray:
    """Return full distribution b(a|s) for epsilon-greedy over argmax(Q)."""
    A = q_values.shape[0]
    probs = np.ones(A, dtype=np.float32) * (eps / A)
    greedy = int(np.argmax(q_values))
    probs[greedy] += (1.0 - eps)
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/logged_behavior.npz")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.2, help="behavior epsilon (fixed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="dqn_policy.pth")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    env_cfg = EnvConfig()
    env = UserRetentionEnv(env_cfg, seed=args.seed)

    train_cfg = TrainConfig()  # only used to instantiate agent dims, etc.
    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, cfg=train_cfg)
    agent.load(args.model)  # your agent likely has load(); if not, I’ll adjust in next step
    agent.policy_net.eval()

    rng = np.random.default_rng(args.seed)

    episodes = {
        "states": [],
        "actions": [],
        "rewards": [],
        "b_probs": [],
        "pi_probs": [],       # placeholder; filled later in analysis_ope.py for any target policy
        "pi_probs_all": [],   # placeholder
        "b_probs_all": [],    # behavior probs full dist for DM/DR if needed
    }

    print(f"Collecting {args.episodes} episodes with behavior epsilon={args.epsilon} ...")

    for ep in range(args.episodes):
        s, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False

        S, A, R = [], [], []
        b_taken, b_all = [], []

        while not done:
            with torch.no_grad():
                st = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
                q = agent.policy_net(st).squeeze(0).detach().cpu().numpy()

            probs = eps_greedy_probs(q, args.epsilon)  # behavior distribution b(a|s)
            a = int(rng.choice(np.arange(env.action_space.n), p=probs))

            s2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            S.append(s)
            A.append(a)
            R.append(float(r))
            b_taken.append(float(probs[a]))
            b_all.append(probs.copy())

            s = s2

        episodes["states"].append(np.asarray(S, dtype=np.float32))
        episodes["actions"].append(np.asarray(A, dtype=np.int64))
        episodes["rewards"].append(np.asarray(R, dtype=np.float32))
        episodes["b_probs"].append(np.asarray(b_taken, dtype=np.float32))
        episodes["b_probs_all"].append(np.asarray(b_all, dtype=np.float32))

    np.savez_compressed(args.out, **{k: np.array(v, dtype=object) for k, v in episodes.items()})
    print(f"Saved behavior log to: {args.out}")


if __name__ == "__main__":
    main()
