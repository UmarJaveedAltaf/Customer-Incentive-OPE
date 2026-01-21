import random
import torch
import torch.nn.functional as F
from torch import optim

from .replay_buffer import ReplayBuffer
from ..models.dqn_network import DuelingDQN


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks (Double + Dueling DQN)
        self.policy_net = DuelingDQN(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

        # Replay buffer
        self.replay = ReplayBuffer(cfg.replay_capacity)

        # Hyperparameters
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size

        # Exploration
        self.epsilon = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay

        self.total_steps = 0

    def select_action(self, state, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection"""
        if (not eval_mode) and (random.random() < self.epsilon):
            return random.randrange(self.action_dim)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(torch.argmax(q_values, dim=1).item())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self) -> float:
        """One Double + Dueling DQN update step"""
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a)
        q_values = self.policy_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # Huber loss
        loss = F.smooth_l1_loss(q_sa, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.total_steps += 1
        self.update_epsilon()

        return float(loss.item())

    def hard_update_target(self):
        """Hard update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # -------------------------
    # ✅ SAVE / LOAD (OPE READY)
    # -------------------------
    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "epsilon": self.epsilon
        }, path)

    def load(self, path: str, eval_mode: bool = True):
        checkpoint = torch.load(path, map_location=self.device)

        # Old-style checkpoint (policy only)
        if isinstance(checkpoint, dict) and "policy_state_dict" not in checkpoint:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
        else:
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]

        if eval_mode:
            self.policy_net.eval()
            self.target_net.eval()
