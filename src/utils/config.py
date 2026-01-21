from dataclasses import dataclass

ACTION_NAMES = [
    "no_action",
    "10_percent_coupon",
    "15_percent_coupon",
    "free_shipping",
    "loyalty_points",
    "bundle_offer",
]

@dataclass
class EnvConfig:
    num_users: int = 80
    episode_length: int = 80              # steps per episode
    total_budget: float = 500.0
    max_days_inactive: int = 30

    # segment distribution: low/med/high
    seg_p_low: float = 0.50
    seg_p_med: float = 0.30
    seg_p_high: float = 0.20

@dataclass
class TrainConfig:
    seed: int = 42

    # DQN core
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 1_000
    target_update_every: int = 250

    # exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay: float = 0.997

    # training
    episodes: int = 800
    max_steps_per_episode: int = 200

    # network
    hidden_dim: int = 128

    # logging
    log_every: int = 20
