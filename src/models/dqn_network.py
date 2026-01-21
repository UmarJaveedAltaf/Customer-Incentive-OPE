import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + A(s,a) - mean(A)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals
