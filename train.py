import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from src.utils.config import EnvConfig, TrainConfig, ACTION_NAMES
from src.utils.seed import seed_everything
from src.environment.user_retention_env import UserRetentionEnv
from src.agents.dqn_agent import DQNAgent


def moving_avg(x, w=50):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    # -----------------------------
    # Config & setup
    # -----------------------------
    tcfg = TrainConfig()
    ecfg = EnvConfig()

    seed_everything(tcfg.seed)

    env = UserRetentionEnv(ecfg, seed=tcfg.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, tcfg)

    rewards_hist = []
    loss_hist = []
    action_hist = np.zeros(action_dim, dtype=np.int64)

    # -----------------------------
    # Training loop
    # -----------------------------
    for ep in tqdm(range(tcfg.episodes)):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_losses = []

        for _ in range(min(tcfg.max_steps_per_episode, ecfg.episode_length)):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay.push(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            action_hist[action] += 1

            # Train after warmup
            if len(agent.replay) > tcfg.warmup_steps:
                loss = agent.train_step()
                if loss > 0:
                    ep_losses.append(loss)

                if agent.total_steps % tcfg.target_update_every == 0:
                    agent.hard_update_target()

                agent.update_epsilon()

            if done:
                break

        rewards_hist.append(ep_reward)
        loss_hist.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if ep % tcfg.log_every == 0 and ep > 0:
            print(f"\nEpisode {ep}")
            print(f"  Reward (avg last {tcfg.log_every}): {np.mean(rewards_hist[-tcfg.log_every:]):.2f}")
            print(f"  Loss   (avg last {tcfg.log_every}): {np.mean(loss_hist[-tcfg.log_every:]):.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")

    # -----------------------------
    # Save trained model 
    # -----------------------------
    agent.save("dqn_policy.pth")
    print("\n Saved trained model to dqn_policy.pth")

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_hist, label="Episode Reward", alpha=0.7)
    plt.plot(
        range(len(moving_avg(rewards_hist))),
        moving_avg(rewards_hist),
        label="Moving Avg",
        linewidth=2,
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(loss_hist, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nAction usage distribution:")
    total = action_hist.sum()
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {i} {name:18s}: {action_hist[i] / max(total, 1):.2%}")


if __name__ == "__main__":
    main()
