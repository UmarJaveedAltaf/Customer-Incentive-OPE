# src/ope/models.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception as e:
    RandomForestRegressor = None


def _ensure_2d_float32(x) -> np.ndarray:
    """Make sure x is a numeric 2D float32 array (handles object arrays)."""
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(arr, axis=0)
        except Exception:
            arr = np.asarray(list(arr))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _ensure_1d_int64(x) -> np.ndarray:
    """Make sure x is a numeric 1D int64 array (handles object arrays)."""
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(arr, axis=0)
        except Exception:
            arr = np.asarray(list(arr))
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    return arr


def _ensure_1d_float32(x) -> np.ndarray:
    """Make sure x is a numeric 1D float32 array (handles object arrays)."""
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(arr, axis=0)
        except Exception:
            arr = np.asarray(list(arr))
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr


@dataclass
class RewardModel:
    """
    Simple supervised reward model r_hat(s, a) ≈ r.

    Features: concat([state, one_hot(action)]) => shape = state_dim + action_dim
    """
    action_dim: int
    seed: int = 0
    n_estimators: int = 150
    max_depth: int | None = 12
    n_jobs: int = -1

    def __post_init__(self):
        if RandomForestRegressor is None:
            raise ImportError(
                "scikit-learn is required for DM/DR reward model. "
                "pip install scikit-learn"
            )
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )

    def _featurize(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        states: [N, state_dim]
        actions: [N]
        returns: [N, state_dim + action_dim]
        """
        states = _ensure_2d_float32(states)
        actions = _ensure_1d_int64(actions)
        N = states.shape[0]

        a_oh = np.zeros((N, self.action_dim), dtype=np.float32)
        a_oh[np.arange(N), actions] = 1.0
        return np.concatenate([states, a_oh], axis=1)

    def fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        states = _ensure_2d_float32(states)
        actions = _ensure_1d_int64(actions)
        rewards = _ensure_1d_float32(rewards)

        X = self._featurize(states, actions)
        self.model.fit(X, rewards)
        return self

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = _ensure_2d_float32(states)
        actions = _ensure_1d_int64(actions)
        X = self._featurize(states, actions)
        return self.model.predict(X).astype(np.float32)

    def predict_all(self, states: np.ndarray) -> np.ndarray:
        """
        Predict r_hat(s, a) for ALL actions at each state.
        returns: [N, action_dim]
        """
        states = _ensure_2d_float32(states)
        N = states.shape[0]
        out = np.zeros((N, self.action_dim), dtype=np.float32)

        for a in range(self.action_dim):
            aa = np.full((N,), a, dtype=np.int64)
            X = self._featurize(states, aa)
            out[:, a] = self.model.predict(X).astype(np.float32)

        return out


def build_default_reward_model(action_dim: int, seed: int = 0) -> RewardModel:
    """
    Factory used by analysis_ope.py
    """
    return RewardModel(action_dim=action_dim, seed=seed)


def flatten_episode_data(episodes: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    episodes contains lists:
      episodes["states"]  -> list of [T_i, state_dim]
      episodes["actions"] -> list of [T_i]
      episodes["rewards"] -> list of [T_i]

    returns:
      flat_states  [sum_T, state_dim]
      flat_actions [sum_T]
      flat_rewards [sum_T]
    """
    all_states = []
    all_actions = []
    all_rewards = []

    for s, a, r in zip(episodes["states"], episodes["actions"], episodes["rewards"]):
        s = _ensure_2d_float32(s)
        a = _ensure_1d_int64(a)
        r = _ensure_1d_float32(r)

        # guard length alignment
        T = min(len(s), len(a), len(r))
        all_states.append(s[:T])
        all_actions.append(a[:T])
        all_rewards.append(r[:T])

    flat_states = np.concatenate(all_states, axis=0).astype(np.float32)
    flat_actions = np.concatenate(all_actions, axis=0).astype(np.int64)
    flat_rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)

    return flat_states, flat_actions, flat_rewards
