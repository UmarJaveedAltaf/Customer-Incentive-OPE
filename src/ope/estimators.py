# src/ope/estimators.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Callable


@dataclass
class OPEResult:
    estimate: float
    ci_low: float
    ci_high: float
    details: Dict


def _bootstrap_ci(values: np.ndarray, n_boot: int = 500, alpha: float = 0.05, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(values[idx].mean())
    boots = np.array(boots, dtype=np.float64)
    lo = np.quantile(boots, alpha / 2.0)
    hi = np.quantile(boots, 1.0 - alpha / 2.0)
    return float(lo), float(hi)


def ips_per_episode(episodes: Dict, eps: float = 1e-12) -> np.ndarray:
    """
    Per-episode IPS estimate:
      V^IPS = sum_t (pi(a_t|s_t)/b(a_t|s_t)) * r_t

    episodes dict keys (arrays):
      - rewards: list of 1D arrays
      - pi_probs: list of 1D arrays  (pi(a_t|s_t) for taken action)
      - b_probs:  list of 1D arrays  (b(a_t|s_t) for taken action)
    """
    vals = []
    for r, pi_p, b_p in zip(episodes["rewards"], episodes["pi_probs"], episodes["b_probs"]):
        w = np.clip(pi_p / (b_p + eps), 0.0, 50.0)  # clip for stability
        vals.append(float(np.sum(w * r)))
    return np.array(vals, dtype=np.float64)


def wis_per_episode(episodes: Dict, eps: float = 1e-12) -> np.ndarray:
    """
    Weighted IPS (self-normalized) per episode:
      V^WIS = (sum_t w_t r_t) / (sum_t w_t)
    """
    vals = []
    for r, pi_p, b_p in zip(episodes["rewards"], episodes["pi_probs"], episodes["b_probs"]):
        w = np.clip(pi_p / (b_p + eps), 0.0, 50.0)
        denom = float(np.sum(w) + eps)
        vals.append(float(np.sum(w * r) / denom))
    return np.array(vals, dtype=np.float64)


def dm_per_episode(episodes: Dict, rhat_fn: Callable[[np.ndarray, int], float]) -> np.ndarray:
    """
    Direct Method: replace true rewards with model predictions:
      V^DM = sum_t E_{a~pi}[ rhat(s_t, a) ]
    We approximate expectation using pi_probs_all(t, a) saved in the dataset.
    """
    vals = []
    for states, pi_all in zip(episodes["states"], episodes["pi_probs_all"]):
        # states: [T, state_dim], pi_all: [T, action_dim]
        T, A = pi_all.shape
        total = 0.0
        for t in range(T):
            s = states[t]
            # expected reward under pi:
            er = 0.0
            for a in range(A):
                er += float(pi_all[t, a]) * float(rhat_fn(s, int(a)))
            total += er
        vals.append(float(total))
    return np.array(vals, dtype=np.float64)


def dr_per_episode(episodes: Dict, qhat_fn: Callable[[np.ndarray, int], float], eps: float = 1e-12) -> np.ndarray:
    """
    Doubly Robust (step-wise):
      V^DR = sum_t [ E_{a~pi} qhat(s_t,a) + w_t (r_t - qhat(s_t,a_t)) ]
      where w_t = pi(a_t|s_t)/b(a_t|s_t)
    """
    vals = []
    for states, actions, rewards, pi_all, pi_taken, b_taken in zip(
        episodes["states"], episodes["actions"], episodes["rewards"],
        episodes["pi_probs_all"], episodes["pi_probs"], episodes["b_probs"]
    ):
        T, A = pi_all.shape
        total = 0.0
        for t in range(T):
            s = states[t]
            # model expectation under pi
            exp_q = 0.0
            for a in range(A):
                exp_q += float(pi_all[t, a]) * float(qhat_fn(s, int(a)))
            w = float(np.clip(pi_taken[t] / (b_taken[t] + eps), 0.0, 50.0))
            a_t = int(actions[t])
            r_t = float(rewards[t])
            total += exp_q + w * (r_t - float(qhat_fn(s, a_t)))
        vals.append(float(total))
    return np.array(vals, dtype=np.float64)


def summarize(values: np.ndarray, n_boot: int = 500, alpha: float = 0.05, seed: int = 0) -> OPEResult:
    est = float(values.mean())
    lo, hi = _bootstrap_ci(values, n_boot=n_boot, alpha=alpha, rng_seed=seed)
    return OPEResult(estimate=est, ci_low=lo, ci_high=hi, details={"n": int(len(values))})
