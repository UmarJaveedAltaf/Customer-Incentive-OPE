import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from src.environment.user_retention_env import UserRetentionEnv
from src.models.dqn_network import DuelingDQN
from src.utils.config import EnvConfig, ACTION_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = UserRetentionEnv(EnvConfig())
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DuelingDQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("dqn_policy.pth", map_location=device))
model.eval()

segment_actions = defaultdict(list)

state, _ = env.reset()
done = False

while not done:
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = int(model(s).argmax(dim=1).item())

    segment = np.argmax(state[:3])  # one-hot encoded segment
    segment_actions[segment].append(action)

    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

labels = ["Low Value", "Medium Value", "High Value"]

plt.figure(figsize=(10, 5))

for seg, actions in segment_actions.items():
    counts = np.bincount(actions, minlength=action_dim)
    plt.bar(
        np.arange(action_dim) + seg * 0.25,
        counts / counts.sum(),
        width=0.25,
        label=labels[seg]
    )

plt.xticks(range(action_dim), ACTION_NAMES, rotation=30)
plt.ylabel("Action Frequency")
plt.title("Segment-wise Incentive Strategy Learned by DQN")
plt.legend()
plt.tight_layout()
plt.show()
