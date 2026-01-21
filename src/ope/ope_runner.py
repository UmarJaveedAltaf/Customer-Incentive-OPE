# src/ope/ope_runner.py
import os
import numpy as np
import torch
from tqdm import tqdm

from src.utils.config import TrainConfig
from src.agents.dqn_agent import DQNAgent
from src.ope.models import build_default_reward_model, flatten_episode_data


def _to_float32_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(arr, axis=0)
        except Exception:
            arr = np.asarray(list(arr))
    return arr.astype(np.float32)


def _to_int64_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(arr, axis=0)
        except Exception:
            arr = np.asarray(list(arr))
    return arr.astype(np.int64)


def greedy_target_probs(agent, states: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    T = len(states)
    pi = np.zeros((T, agent.action_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch = states[i:i + batch_size]
            st = torch.tensor(batch, dtype=torch.float32, device=agent.device)
            q = agent.policy_net(st)
            greedy = q.argmax(dim=1).cpu().numpy()
            pi[np.arange(i, i + len(batch)), greedy] = 1.0
    return pi


def _ips_episode(rewards, pi_taken, b_taken, clip_rho=50.0):
    b_taken = np.clip(b_taken, 1e-8, 1.0)
    rho = np.clip(pi_taken / b_taken, 0.0, clip_rho)
    return float(np.sum(rho * rewards))


def _wis_episode(rewards, pi_taken, b_taken, clip_rho=50.0):
    b_taken = np.clip(b_taken, 1e-8, 1.0)
    rho = np.clip(pi_taken / b_taken, 0.0, clip_rho)
    denom = float(np.sum(rho)) + 1e-8
    w = rho / denom
    return float(np.sum(w * rewards))


def _dm_episode(pi_all, rhat_all):
    exp_r = np.sum(pi_all * rhat_all, axis=1)
    return float(np.sum(exp_r))


def _dr_episode(actions, rewards, pi_all, pi_taken, b_taken, qhat_all, clip_rho=50.0):
    b_taken = np.clip(b_taken, 1e-8, 1.0)
    rho = np.clip(pi_taken / b_taken, 0.0, clip_rho)

    exp_q = np.sum(pi_all * qhat_all, axis=1)
    q_taken = qhat_all[np.arange(len(actions)), actions]

    return float(np.sum(exp_q + rho * (rewards - q_taken)))


def bootstrap_ci(values: np.ndarray, n_boot=800, seed=0):
    vals = np.asarray(values, dtype=np.float32)
    n = len(vals)
    rng = np.random.default_rng(seed)

    est = float(vals.mean()) if n else float("nan")
    boots = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = vals[idx].mean()

    return {
        "estimate": est,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": n,
        "n_boot": n_boot,
    }


def run_ope(
    log_path: str = "data/logged_behavior.npz",
    model_path: str = "dqn_policy.pth",
    n_boot: int = 800,
    clip_rho: float = 50.0,
    progress_cb=None,   # callable(step_idx:int, total:int, msg:str)
):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Logged data not found at: {log_path}")

    data = np.load(log_path, allow_pickle=True)
    episodes = {k: list(data[k]) for k in data.files}

    first_states = _to_float32_array(episodes["states"][0])
    first_b_all = _to_float32_array(episodes["b_probs_all"][0])
    state_dim = first_states.shape[1]
    action_dim = first_b_all.shape[1]

    cfg = TrainConfig()
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)
    agent.load(model_path)
    agent.policy_net.eval()

    # Step 1: normalize + compute pi
    if progress_cb:
        progress_cb(1, 4, "Step 1/4: Computing target policy probabilities...")

    norm_states, norm_actions, norm_rewards, norm_b_probs, norm_b_probs_all = [], [], [], [], []
    pi_probs, pi_probs_all = [], []

    n_eps = len(episodes["states"])
    for i in tqdm(range(n_eps), desc="OPE Step 1 (pi)"):
        states = _to_float32_array(episodes["states"][i])
        actions = _to_int64_array(episodes["actions"][i]).reshape(-1)
        rewards = _to_float32_array(episodes["rewards"][i]).reshape(-1)
        b_taken = _to_float32_array(episodes["b_probs"][i]).reshape(-1)
        b_all = _to_float32_array(episodes["b_probs_all"][i])

        pi_all = greedy_target_probs(agent, states)
        pi_taken = pi_all[np.arange(len(actions)), actions]

        norm_states.append(states)
        norm_actions.append(actions)
        norm_rewards.append(rewards)
        norm_b_probs.append(b_taken)
        norm_b_probs_all.append(b_all)
        pi_probs_all.append(pi_all.astype(np.float32))
        pi_probs.append(pi_taken.astype(np.float32))

    episodes["states"] = norm_states
    episodes["actions"] = norm_actions
    episodes["rewards"] = norm_rewards
    episodes["b_probs"] = norm_b_probs
    episodes["b_probs_all"] = norm_b_probs_all
    episodes["pi_probs_all"] = pi_probs_all
    episodes["pi_probs"] = pi_probs

    # Step 2: IPS / WIS
    if progress_cb:
        progress_cb(2, 4, "Step 2/4: Computing IPS / WIS...")

    ips_vals, wis_vals = [], []
    for i in tqdm(range(n_eps), desc="OPE Step 2 (IPS/WIS)"):
        ips_vals.append(_ips_episode(episodes["rewards"][i], episodes["pi_probs"][i], episodes["b_probs"][i], clip_rho))
        wis_vals.append(_wis_episode(episodes["rewards"][i], episodes["pi_probs"][i], episodes["b_probs"][i], clip_rho))

    # Step 3: reward model (DM/DR)
    if progress_cb:
        progress_cb(3, 4, "Step 3/4: Training reward model (DM/DR)...")

    flat_states, flat_actions, flat_rewards = flatten_episode_data(episodes)
    rmodel = build_default_reward_model(action_dim=action_dim, seed=0)
    rmodel.fit(flat_states, flat_actions, flat_rewards)

    has_predict_all = hasattr(rmodel, "predict_all")

    def predict_all_actions_fallback(states: np.ndarray):
        T = states.shape[0]
        out = np.zeros((T, action_dim), dtype=np.float32)
        for a in range(action_dim):
            aa = np.full((T,), a, dtype=np.int64)
            out[:, a] = rmodel.predict(states, aa).astype(np.float32)
        return out

    # Step 4: DM / DR
    if progress_cb:
        progress_cb(4, 4, "Step 4/4: Computing DM / DR...")

    dm_vals, dr_vals = [], []
    for i in tqdm(range(n_eps), desc="OPE Step 4 (DM/DR)"):
        states = episodes["states"][i]
        actions = episodes["actions"][i]
        rewards = episodes["rewards"][i]
        b_taken = episodes["b_probs"][i]
        pi_taken = episodes["pi_probs"][i]
        pi_all = episodes["pi_probs_all"][i]

        rhat_all = rmodel.predict_all(states).astype(np.float32) if has_predict_all else predict_all_actions_fallback(states)
        qhat_all = rhat_all

        dm_vals.append(_dm_episode(pi_all, rhat_all))
        dr_vals.append(_dr_episode(actions, rewards, pi_all, pi_taken, b_taken, qhat_all, clip_rho))

    # Bootstrap CIs
    if progress_cb:
        progress_cb(4, 4, "Bootstrapping confidence intervals...")

    res = {
        "per_episode": {
            "IPS": np.array(ips_vals, dtype=np.float32),
            "WIS": np.array(wis_vals, dtype=np.float32),
            "DM": np.array(dm_vals, dtype=np.float32),
            "DR": np.array(dr_vals, dtype=np.float32),
        },
        "summary": {
            "IPS": bootstrap_ci(ips_vals, n_boot=n_boot, seed=0),
            "WIS": bootstrap_ci(wis_vals, n_boot=n_boot, seed=1),
            "DM": bootstrap_ci(dm_vals, n_boot=n_boot, seed=2),
            "DR": bootstrap_ci(dr_vals, n_boot=n_boot, seed=3),
        },
        "meta": {
            "n_episodes": n_eps,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "clip_rho": clip_rho,
            "n_boot": n_boot,
        },
    }
    return res
