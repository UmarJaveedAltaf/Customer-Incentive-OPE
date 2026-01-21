import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .competitor import CompetitorPolicy
from .user_simulator import UserSimulator, UserProfile
from ..utils.config import ACTION_NAMES, EnvConfig


class UserRetentionEnv(gym.Env):
    """
    Observation: 10 floats in [0,1]
      0-2: segment one-hot (low/med/high)
      3: days_inactive normalized
      4: engagement score
      5: cart_abandon score
      6: budget_remaining normalized
      7: promo_fatigue score
      8: competitor_offer_strength normalized (0.0 -> 0%, 0.666.. -> 10%, 1.0 -> 15%)
      9: competitor_active_flag (0/1)

    Action: 0..5 (incentive type)

    Reward:
      + revenue (after discount) if purchase
      - incentive cost
      - churn penalty (normalized LTV)
      + retention credit if incentive used and no churn
      - budget overspend penalty (soft)
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        # UPDATED: obs dim 10
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Action costs in dollars (marketing cost)
        self.action_cost = np.array([0.0, 5.0, 7.5, 4.0, 2.5, 6.0], dtype=np.float32)
        # Discount multipliers on revenue
        self.discount_mult = np.array([1.0, 0.90, 0.85, 1.0, 1.0, 1.0], dtype=np.float32)

        # Competitor policy
        self.competitor = CompetitorPolicy()
        self.competitor_offer = 0.0  # 0.0, 0.10, 0.15

        # internal episode state
        self.users = []
        self.user_states = []
        self.idx = 0
        self.steps = 0
        self.budget = cfg.total_budget

    def _make_user_state(self, segment: int):
        return {
            "segment": segment,
            "days_inactive": int(self.rng.integers(0, self.cfg.max_days_inactive + 1)),
            "engagement": float(np.clip(self.rng.normal(0.55, 0.2), 0.0, 1.0)),
            "cart_abandon": float(np.clip(self.rng.normal(0.35, 0.25), 0.0, 1.0)),
            "promo_fatigue": float(np.clip(self.rng.normal(0.2, 0.15), 0.0, 1.0)),
        }

    def _segment_name(self, seg: int) -> str:
        if seg == 2:
            return "high"
        if seg == 1:
            return "medium"
        return "low"

    def _normalize_comp_offer(self, offer: float) -> float:
        """
        Normalize competitor offer to [0,1] for {0.0, 0.10, 0.15}
        0.00 -> 0.0
        0.10 -> 0.666...
        0.15 -> 1.0
        """
        return float(np.clip(offer / 0.15, 0.0, 1.0))

    def _obs_from_state(self, st: dict) -> np.ndarray:
        seg = st["segment"]
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[seg] = 1.0

        days_norm = np.float32(np.clip(st["days_inactive"] / self.cfg.max_days_inactive, 0.0, 1.0))
        bud_norm = np.float32(np.clip(self.budget / self.cfg.total_budget, 0.0, 1.0))

        comp_norm = np.float32(self._normalize_comp_offer(self.competitor_offer))
        comp_flag = np.float32(1.0 if self.competitor_offer > 0.0 else 0.0)

        # UPDATED: return 10-dim obs
        obs = np.array(
            [
                one_hot[0],
                one_hot[1],
                one_hot[2],
                days_norm,
                np.float32(st["engagement"]),
                np.float32(st["cart_abandon"]),
                bud_norm,
                np.float32(st["promo_fatigue"]),
                comp_norm,
                comp_flag,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.budget = float(self.cfg.total_budget)
        self.steps = 0
        self.idx = 0

        # Reset competitor
        self.competitor_offer = 0.0

        # Create cohort
        self.users = []
        self.user_states = []

        probs = [self.cfg.seg_p_low, self.cfg.seg_p_med, self.cfg.seg_p_high]
        for uid in range(self.cfg.num_users):
            seg = int(self.rng.choice([0, 1, 2], p=probs))
            profile = UserProfile(user_id=uid, segment=seg)
            sim = UserSimulator(profile, self.rng)
            st = self._make_user_state(seg)
            self.users.append(sim)
            self.user_states.append(st)

        # Shuffle user contact order
        perm = self.rng.permutation(self.cfg.num_users)
        self.users = [self.users[i] for i in perm]
        self.user_states = [self.user_states[i] for i in perm]

        # Initialize competitor offer for first user
        first_st = self.user_states[self.idx]
        seg_name = self._segment_name(first_st["segment"])
        self.competitor_offer = float(self.competitor.select_offer(seg_name, first_st["days_inactive"]))

        obs = self._obs_from_state(first_st)
        return obs, {}

    def step(self, action: int):
        st = self.user_states[self.idx]
        sim = self.users[self.idx]

        # Update competitor offer based on current user state
        seg_name = self._segment_name(st["segment"])
        self.competitor_offer = float(self.competitor.select_offer(seg_name, st["days_inactive"]))

        # Budget constraint
        cost = float(self.action_cost[action])
        if self.budget < cost:
            action = 0
            cost = 0.0

        # Promo fatigue dynamics
        if action != 0:
            st["promo_fatigue"] = float(np.clip(st["promo_fatigue"] + 0.08, 0.0, 1.0))
        else:
            st["promo_fatigue"] = float(np.clip(st["promo_fatigue"] - 0.03, 0.0, 1.0))

        # --- Competitor-aware simulation ---
        # If your UserSimulator.respond() supports competitor_offer, we pass it.
        # Otherwise we fallback safely to the old signature (no crash).
        try:
            will_purchase, will_churn, order_value, ltv = sim.respond(
                days_inactive=st["days_inactive"],
                engagement=st["engagement"],
                cart_abandon=st["cart_abandon"],
                promo_fatigue=st["promo_fatigue"],
                action=action,
                competitor_offer=self.competitor_offer,  # NEW
            )
        except TypeError:
            will_purchase, will_churn, order_value, ltv = sim.respond(
                days_inactive=st["days_inactive"],
                engagement=st["engagement"],
                cart_abandon=st["cart_abandon"],
                promo_fatigue=st["promo_fatigue"],
                action=action,
            )

        # Revenue
        revenue = 0.0
        if will_purchase:
            revenue = float(order_value * self.discount_mult[action])

        # ----------------------------
        # Balanced reward shaping (unchanged)
        # ----------------------------
        reward = 0.0
        reward += revenue
        reward -= cost

        # Normalize LTV
        ltv_norm = ltv / 1000.0  # high-value ≈ 2.2

        if will_churn:
            reward -= 10.0 * ltv_norm
        else:
            if action != 0:
                reward += 2.0 * ltv_norm

        # Budget pacing penalty
        bud_ratio = self.budget / self.cfg.total_budget
        reward -= (1.0 - bud_ratio) * 0.01 * cost

        # Update budget
        self.budget -= cost
        self.budget = max(self.budget, 0.0)

        # Update inactivity
        if will_purchase:
            st["days_inactive"] = 0
        else:
            st["days_inactive"] = int(min(st["days_inactive"] + 1, self.cfg.max_days_inactive))

        # Step forward
        self.idx += 1
        self.steps += 1

        terminated = (self.steps >= self.cfg.episode_length or self.idx >= self.cfg.num_users)
        truncated = False

        if not terminated:
            # Update competitor offer for NEXT user (so state reflects next interaction)
            next_st = self.user_states[self.idx]
            next_seg_name = self._segment_name(next_st["segment"])
            self.competitor_offer = float(self.competitor.select_offer(next_seg_name, next_st["days_inactive"]))
            obs = self._obs_from_state(next_st)
        else:
            obs = np.zeros((10,), dtype=np.float32)

        info = {
            "action": int(action),
            "cost": float(cost),
            "revenue": float(revenue),
            "purchase": bool(will_purchase),
            "churn": bool(will_churn),
            "segment": int(st["segment"]),
            "budget_remaining": float(self.budget),
            "ltv": float(ltv),
            # NEW info fields
            "competitor_offer": float(self.competitor_offer),
            "competitor_active": bool(self.competitor_offer > 0.0),
        }

        return obs, float(reward), terminated, truncated, info
