import torch
import matplotlib.pyplot as plt

from src.environment.user_retention_env import UserRetentionEnv
from src.models.dqn_network import DuelingDQN
from src.utils.config import EnvConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = UserRetentionEnv(EnvConfig())
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DuelingDQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("dqn_policy.pth", map_location=device))
model.eval()

budgets = []

state, _ = env.reset()
done = False

while not done:
    # 🔒 Robust budget access
    if hasattr(env, "budget"):
        budgets.append(env.budget)
    elif hasattr(env, "current_budget"):
        budgets.append(env.current_budget)
    elif hasattr(env, "remaining_budget"):
        budgets.append(env.remaining_budget)
    else:
        raise AttributeError("No budget attribute found in environment")

    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = int(model(s).argmax(dim=1).item())

    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

plt.figure(figsize=(8, 4))
plt.plot(budgets)
plt.xlabel("Interaction Step")
plt.ylabel("Remaining Marketing Budget")
plt.title("Budget Pacing Learned by DQN")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
