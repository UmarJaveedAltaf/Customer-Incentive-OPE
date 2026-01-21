import numpy as np
import torch

from src.utils.config import EnvConfig, TrainConfig, ACTION_NAMES
from src.utils.seed import seed_everything
from src.environment.user_retention_env import UserRetentionEnv
from src.models.dqn_network import DuelingDQN


def run_policy(env, policy_fn, episodes=50):
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        scores.append(total_reward)

    return float(np.mean(scores)), float(np.std(scores))


def main():
    cfg = TrainConfig()
    env_cfg = EnvConfig()

    seed_everything(cfg.seed)
    env = UserRetentionEnv(env_cfg, seed=cfg.seed)

    # --------------------
    # Baseline policies
    # --------------------
    random_policy = lambda s: np.random.randint(len(ACTION_NAMES))
    no_action_policy = lambda s: 0

    def rule_based_policy(s):
        days_inactive = s[3]
        cart_abandon = s[5]
        high_value = s[2] > 0.5

        if days_inactive > 0.7 and high_value:
            return 2  # 15% coupon
        if days_inactive > 0.7:
            return 1  # 10% coupon
        if cart_abandon > 0.7:
            return 3  # free shipping
        return 0

    print("\nBaseline results:")
    for name, policy in [
        ("Random", random_policy),
        ("No Action", no_action_policy),
        ("Rule-based", rule_based_policy),
    ]:
        mean, std = run_policy(env, policy)
        print(f"{name:20s} mean={mean:9.2f}  std={std:8.2f}")

    # --------------------
    # Dueling + Double DQN
    # --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DuelingDQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load("dqn_policy.pth", map_location=device))
    model.eval()

    def dqn_policy(state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return int(model(s).argmax(dim=1).item())

    mean, std = run_policy(env, dqn_policy)
    print(f"\nDQN Agent           mean={mean:9.2f}  std={std:8.2f}")


if __name__ == "__main__":
    main()
