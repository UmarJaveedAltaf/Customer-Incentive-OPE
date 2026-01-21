print(">>> analysis_ab_test.py STARTED <<<")

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.environment.user_retention_env import UserRetentionEnv
from src.models.dqn_network import DuelingDQN
from src.utils.config import EnvConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPISODES = 20  # small for debug

print("Setting up environment...")
env = UserRetentionEnv(EnvConfig())

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print("Loading DQN model...")
dqn = DuelingDQN(state_dim, action_dim).to(DEVICE)
dqn.load_state_dict(torch.load("dqn_policy.pth", map_location=DEVICE))
dqn.eval()

def dqn_policy(state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(dqn(s).argmax(dim=1).item())

def rule_based_policy(state):
    days_inactive = state[3]
    cart_abandon = state[5]
    high_value = state[2] > 0.5

    if days_inactive > 0.7 and high_value:
        return 2
    if days_inactive > 0.7:
        return 1
    if cart_abandon > 0.7:
        return 3
    return 0

def run_policy(name, policy_fn):
    rewards = []

    print(f"Running {name} policy...")
    for ep in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        print(f"{name} | Episode {ep} reward = {total_reward:.2f}")

    return np.array(rewards)

rule_rewards = run_policy("RULE", rule_based_policy)
dqn_rewards = run_policy("DQN", dqn_policy)

print("\nSUMMARY")
print(f"Rule mean: {rule_rewards.mean():.2f}")
print(f"DQN  mean: {dqn_rewards.mean():.2f}")

print("Plotting results...")

plt.figure(figsize=(8, 4))
sns.boxplot(data=[rule_rewards, dqn_rewards])
plt.xticks([0, 1], ["Rule-based", "DQN"])
plt.title("A/B Test Reward Distribution")
plt.ylabel("Total Reward")
plt.show()

print(">>> analysis_ab_test.py FINISHED <<<")
