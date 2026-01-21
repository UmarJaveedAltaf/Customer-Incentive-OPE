# analysis_ope.py
import os
import numpy as np
import torch
from tqdm import tqdm

from src.utils.config import TrainConfig
from src.agents.dqn_agent import DQNAgent
from src.ope.models import build_default_reward_model, flatten_episode_data


# =========================================================
# Utilities
# =========================================================
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
    """
    Greedy deterministic target policy π(a|s) as one-hot
    Output shape: [T, action_dim]
    """
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


def get_segment_masks(states: np.ndarray):
    """
    Segment encoding:
    states[:, 0:3] = one-hot [low, medium, high]
    Returns boolean masks per segment.
    """
    return {
        "Low": states[:, 0] == 1,
        "Medium": states[:, 1] == 1,
        "High": states[:, 2] == 1,
    }


def segment_id_from_state(state_row: np.ndarray) -> int:
    """
    Returns segment id from one-hot at indices [0,1,2].
    0=Low, 1=Medium, 2=High. If ambiguous, returns argmax.
    """
    seg = int(np.argmax(state_row[:3]))
    return seg


# =========================================================
# Bootstrap CI
# =========================================================
def bootstrap_mean_ci(values: np.ndarray, n_boot=800, seed=0, show_tqdm=True, desc="Bootstrap"):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float32)
    n = len(values)

    estimate = float(values.mean()) if n > 0 else float("nan")
    boots = np.empty(n_boot, dtype=np.float32)

    it = range(n_boot)
    if show_tqdm:
        it = tqdm(it, desc=desc)

    for i in it:
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()

    ci_low = float(np.percentile(boots, 2.5)) if n > 0 else float("nan")
    ci_high = float(np.percentile(boots, 97.5)) if n > 0 else float("nan")

    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "details": {"n": n, "n_boot": n_boot},
        "samples": values,  # per-episode samples (for histograms/dashboard)
    }


# =========================================================
# OPE estimators (episode-wise)
# =========================================================
def ips_episode(rewards, pi_taken, b_taken, clip_rho=50.0):
    rho = np.clip(pi_taken / np.clip(b_taken, 1e-8, 1.0), 0.0, clip_rho)
    return float(np.sum(rho * rewards))


def wis_episode(rewards, pi_taken, b_taken, clip_rho=50.0):
    rho = np.clip(pi_taken / np.clip(b_taken, 1e-8, 1.0), 0.0, clip_rho)
    w = rho / (np.sum(rho) + 1e-8)
    return float(np.sum(w * rewards))


def dm_episode(pi_all, rhat_all):
    # sum_t E_pi[rhat(s_t, a)]
    return float(np.sum(np.sum(pi_all * rhat_all, axis=1)))


def dr_episode(actions, rewards, pi_all, pi_taken, b_taken, rhat_all, clip_rho=50.0):
    # step-wise DR: sum_t [ E_pi[qhat] + rho_t*(r_t - qhat(a_t)) ]
    rho = np.clip(pi_taken / np.clip(b_taken, 1e-8, 1.0), 0.0, clip_rho)
    exp_q = np.sum(pi_all * rhat_all, axis=1)  # greedy one-hot => qhat(greedy)
    q_taken = rhat_all[np.arange(len(actions)), actions]
    return float(np.sum(exp_q + rho * (rewards - q_taken)))


# =========================================================
# Policy gate + rollout recommendations
# =========================================================
def gate_decision(ci_low: float, baseline: float = 0.0, margin: float = 0.0) -> bool:
    """
    PASS if lower CI bound beats baseline by at least margin.
    """
    return (ci_low >= (baseline + margin))


def rollout_recommendations(segment_ci: dict, baseline: float, margin: float = 0.0):
    """
    Returns human-readable partial rollout suggestion per segment.
    """
    recs = {}
    for seg, res in segment_ci.items():
        if res["details"]["n"] == 0:
            recs[seg] = {"decision": "NO DATA", "rollout": "0%", "reason": "No samples in this segment."}
            continue

        passed = gate_decision(res["ci_low"], baseline=baseline, margin=margin)
        if passed:
            uplift_lb = res["ci_low"] - baseline
            if uplift_lb > 200:
                rollout = "50%"
            elif uplift_lb > 100:
                rollout = "25%"
            else:
                rollout = "10%"
            recs[seg] = {
                "decision": "PASS",
                "rollout": rollout,
                "reason": f"CI low ({res['ci_low']:.2f}) ≥ baseline+margin.",
            }
        else:
            recs[seg] = {
                "decision": "HOLD",
                "rollout": "0%",
                "reason": f"CI low ({res['ci_low']:.2f}) < baseline+margin.",
            }
    return recs


# =========================================================
# Time-based rollout simulation (segment-wise DR gate)
# =========================================================
SEG_NAMES = {0: "Low", 1: "Medium", 2: "High"}
SEG_ID = {"Low": 0, "Medium": 1, "High": 2}


def simulate_time_rollout(
    dr_by_segment: dict,
    baseline: float,
    margin: float,
    episodes_per_day: int = 20,
    max_days: int = 30,
    start_segments=("Medium", "High"),
    clip_min_n: int = 5,
    seed: int = 0,
):
    """
    Simulates accumulating more offline evidence (episode DR samples) over time,
    and applying the segment-wise DR CI gate each day.

    - Uses NORMAL-approx CI for speed (monitoring-style):
        mean ± 1.96 * std/sqrt(n)

    Returns:
      timeline_df: rows per (day, segment)
      first_pass_day: dict segment -> day or None
      final_recs: dict segment -> EXPAND/HOLD/NOT_STARTED
    """
    gate = float(baseline + margin)
    rng = np.random.default_rng(seed)

    # Prepare shuffled samples per segment
    seg_samples = {}
    for seg_name in ("Low", "Medium", "High"):
        x = np.asarray(dr_by_segment.get(seg_name, []), dtype=np.float32).copy()
        rng.shuffle(x)
        seg_samples[seg_name] = x

    active = {s: (s in start_segments) for s in ("Low", "Medium", "High")}
    used = {s: 0 for s in ("Low", "Medium", "High")}
    first_pass_day = {s: None for s in ("Low", "Medium", "High")}

    rows = []

    def normal_ci(arr: np.ndarray):
        n = len(arr)
        if n == 0:
            return np.nan, np.nan, np.nan, 0
        m = float(arr.mean())
        if n < 2:
            return m, m, m, n
        s = float(arr.std(ddof=1))
        half = 1.96 * (s / np.sqrt(n))
        return m, m - half, m + half, n

    for day in range(1, max_days + 1):
        # compute each segment's gate status
        for seg_name in ("Low", "Medium", "High"):
            if not active[seg_name]:
                rows.append({
                    "day": day,
                    "segment": seg_name,
                    "segment_id": SEG_ID[seg_name],
                    "n": used[seg_name],
                    "mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "gate": gate,
                    "status": "NOT_STARTED",
                })
                continue

            x = seg_samples[seg_name]
            take = min(episodes_per_day, max(0, len(x) - used[seg_name]))
            used[seg_name] += take

            cur = x[:used[seg_name]]
            mean, lo, hi, n = normal_ci(cur)

            status = "HOLD"
            if n >= clip_min_n and (lo >= gate):
                status = "PASS"
                if first_pass_day[seg_name] is None:
                    first_pass_day[seg_name] = day

            rows.append({
                "day": day,
                "segment": seg_name,
                "segment_id": SEG_ID[seg_name],
                "n": n,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "gate": gate,
                "status": status,
            })

        # simple auto-expansion rule:
        # if Medium and High PASS on this day, start Low from next day (if not started)
        med_today = next(r for r in rows if r["day"] == day and r["segment"] == "Medium")["status"]
        high_today = next(r for r in rows if r["day"] == day and r["segment"] == "High")["status"]
        if (med_today == "PASS") and (high_today == "PASS") and (not active["Low"]):
            active["Low"] = True

    # final recommendations
    last_day = max_days
    final_recs = {}
    for seg_name in ("Low", "Medium", "High"):
        row = [r for r in rows if r["day"] == last_day and r["segment"] == seg_name][0]
        if row["status"] == "PASS":
            final_recs[seg_name] = "EXPAND ✅"
        elif row["status"] == "HOLD":
            final_recs[seg_name] = "HOLD / PAUSE ⚠️"
        else:
            final_recs[seg_name] = "NOT STARTED ⏳"

    import pandas as pd
    timeline_df = pd.DataFrame(rows)
    return timeline_df, first_pass_day, final_recs


# =========================================================
# Core runner (used by CLI + Streamlit)
# =========================================================
def run_ope(
    log_path="data/logged_behavior.npz",
    model_path="dqn_policy.pth",
    n_boot=800,
    baseline=0.0,
    margin=0.0,
    clip_rho=50.0,
    quiet=False,
):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Logged data not found: {log_path}")

    data = np.load(log_path, allow_pickle=True)
    episodes = {k: list(data[k]) for k in data.files}

    # Pre-allocate storage to avoid list-assignment issues
    n_eps = len(episodes.get("states", []))
    if n_eps == 0:
        raise ValueError("No episodes found in logged data (states is empty).")

    episodes["pi_probs"] = [None] * n_eps
    episodes["pi_probs_all"] = [None] * n_eps

    # Infer dims
    state_dim = _to_float32_array(episodes["states"][0]).shape[1]
    action_dim = _to_float32_array(episodes["b_probs_all"][0]).shape[1]

    # Load policy
    cfg = TrainConfig()
    agent = DQNAgent(state_dim, action_dim, cfg)
    agent.load(model_path)
    agent.policy_net.eval()

    # -------- Step 1 --------
    if not quiet:
        print("Step 1/4: Computing target policy probabilities...")
    for i in tqdm(range(n_eps), desc="Episodes", disable=quiet):
        states = _to_float32_array(episodes["states"][i])
        actions = _to_int64_array(episodes["actions"][i]).reshape(-1)
        rewards = _to_float32_array(episodes["rewards"][i]).reshape(-1)
        b_taken = _to_float32_array(episodes["b_probs"][i]).reshape(-1)

        pi_all = greedy_target_probs(agent, states)
        pi_taken = pi_all[np.arange(len(actions)), actions]

        episodes["states"][i] = states
        episodes["actions"][i] = actions
        episodes["rewards"][i] = rewards
        episodes["b_probs"][i] = b_taken
        episodes["pi_probs_all"][i] = pi_all
        episodes["pi_probs"][i] = pi_taken

    # -------- Step 2 --------
    if not quiet:
        print("Step 2/4: Computing IPS / WIS...")
    ips_vals, wis_vals = [], []
    for i in tqdm(range(n_eps), desc="IPS/WIS", disable=quiet):
        ips_vals.append(
            ips_episode(
                episodes["rewards"][i],
                episodes["pi_probs"][i],
                episodes["b_probs"][i],
                clip_rho=clip_rho,
            )
        )
        wis_vals.append(
            wis_episode(
                episodes["rewards"][i],
                episodes["pi_probs"][i],
                episodes["b_probs"][i],
                clip_rho=clip_rho,
            )
        )

    # -------- Step 3 --------
    if not quiet:
        print("Step 3/4: Training reward model...")
    flat_states, flat_actions, flat_rewards = flatten_episode_data(episodes)
    rmodel = build_default_reward_model(action_dim, seed=0)
    rmodel.fit(flat_states, flat_actions, flat_rewards)

    def predict_all_actions(states: np.ndarray):
        T = states.shape[0]
        out = np.zeros((T, action_dim), dtype=np.float32)
        for a in range(action_dim):
            out[:, a] = rmodel.predict(states, np.full(T, a, dtype=np.int64)).astype(np.float32)
        return out

    # -------- Step 4 --------
    if not quiet:
        print("Step 4/4: Computing DM / DR + Segment-wise DR...")
    dm_vals, dr_vals = [], []
    segment_dr = {"Low": [], "Medium": [], "High": []}
    segments_episode = []  # per-episode segment id inferred from first state row

    for i in tqdm(range(n_eps), desc="DM/DR", disable=quiet):
        states = episodes["states"][i]
        actions = episodes["actions"][i]
        rewards = episodes["rewards"][i]
        b_taken = episodes["b_probs"][i]
        pi_taken = episodes["pi_probs"][i]
        pi_all = episodes["pi_probs_all"][i]

        # store episode segment id (for dashboard)
        seg_id = segment_id_from_state(states[0])
        segments_episode.append(seg_id)

        rhat_all = predict_all_actions(states)

        dm_vals.append(dm_episode(pi_all, rhat_all))
        dr_i = dr_episode(actions, rewards, pi_all, pi_taken, b_taken, rhat_all, clip_rho=clip_rho)
        dr_vals.append(dr_i)

        # step-level segment masks (within the episode)
        masks = get_segment_masks(states)
        for seg, mask in masks.items():
            if int(mask.sum()) == 0:
                continue
            segment_dr[seg].append(
                dr_episode(
                    actions[mask],
                    rewards[mask],
                    pi_all[mask],
                    pi_taken[mask],
                    b_taken[mask],
                    rhat_all[mask],
                    clip_rho=clip_rho,
                )
            )

    # -------- Bootstrap CIs --------
    if not quiet:
        print("Bootstrapping confidence intervals...")
    res_ips = bootstrap_mean_ci(np.array(ips_vals), n_boot=n_boot, seed=0, show_tqdm=not quiet, desc="Bootstrap IPS")
    res_wis = bootstrap_mean_ci(np.array(wis_vals), n_boot=n_boot, seed=1, show_tqdm=not quiet, desc="Bootstrap WIS")
    res_dm = bootstrap_mean_ci(np.array(dm_vals), n_boot=n_boot, seed=2, show_tqdm=not quiet, desc="Bootstrap DM")
    res_dr = bootstrap_mean_ci(np.array(dr_vals), n_boot=n_boot, seed=3, show_tqdm=not quiet, desc="Bootstrap DR")

    segment_ci = {
        seg: bootstrap_mean_ci(
            np.array(vals, dtype=np.float32),
            n_boot=n_boot,
            seed=10 + j,
            show_tqdm=not quiet,
            desc=f"Bootstrap DR {seg}",
        )
        for j, (seg, vals) in enumerate(segment_dr.items())
    }

    # -------- Gate + Rollout --------
    gate = {
        seg: {
            "pass": gate_decision(res["ci_low"], baseline=baseline, margin=margin),
            "baseline": float(baseline),
            "margin": float(margin),
        }
        for seg, res in segment_ci.items()
    }
    recs = rollout_recommendations(segment_ci, baseline=baseline, margin=margin)

    # Add explicit gate fields for dashboard convenience
    gate_value = float(baseline + margin)

    results = {
        "overall": {"IPS": res_ips, "WIS": res_wis, "DM": res_dm, "DR": res_dr},
        "segment": segment_ci,
        "gate": gate,
        "rollout": recs,
        "meta": {
            "episodes": n_eps,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "clip_rho": float(clip_rho),
            "baseline": float(baseline),
            "margin": float(margin),
            "gate_value": gate_value,
            "n_boot": int(n_boot),
            "log_path": log_path,
            "model_path": model_path,
        },

        # ---- For dashboard visualizations / overlays ----
        "episode_samples": {
            "IPS": np.array(ips_vals, dtype=np.float32),
            "WIS": np.array(wis_vals, dtype=np.float32),
            "DM": np.array(dm_vals, dtype=np.float32),
            "DR": np.array(dr_vals, dtype=np.float32),
        },
        "segments_episode": np.array(segments_episode, dtype=np.int64),  # 0/1/2 per episode
        "dr_by_segment": {
            "Low": np.array(segment_dr["Low"], dtype=np.float32),
            "Medium": np.array(segment_dr["Medium"], dtype=np.float32),
            "High": np.array(segment_dr["High"], dtype=np.float32),
        },

        # Expose rollout simulator as data (dashboard can call simulate_time_rollout itself too)
        "rollout_sim": {
            "available": True,
            "note": "Use simulate_time_rollout(dr_by_segment, baseline, margin, ...) for time-based rollout.",
        },
    }
    return results


def main():
    results = run_ope(
        log_path="data/logged_behavior.npz",
        model_path="dqn_policy.pth",
        n_boot=800,
        baseline=0.0,
        margin=0.0,
        clip_rho=50.0,
        quiet=False,
    )

    print("\n=== OFF-POLICY EVALUATION (Greedy DQN) ===")
    for k, v in results["overall"].items():
        print(f"{k:<4}: {v['estimate']:8.2f}  CI [{v['ci_low']:8.2f}, {v['ci_high']:8.2f}]  n={v['details']['n']}")

    print("\n=== SEGMENT-WISE DR GATE ===")
    for seg, res in results["segment"].items():
        decision = "PASS ✅" if results["gate"][seg]["pass"] else "HOLD ⚠️"
        print(f"{seg:<6} DR: {res['estimate']:8.2f}  CI [{res['ci_low']:8.2f}, {res['ci_high']:8.2f}]  → {decision}")

    print("\n=== PARTIAL ROLLOUT RECOMMENDATIONS ===")
    for seg, rec in results["rollout"].items():
        print(f"{seg:<6} -> {rec['rollout']:<3} | {rec['decision']:<7} | {rec['reason']}")

    # Optional: show a quick rollout simulation in CLI
    # (Dashboard will do this interactively)
    try:
        timeline_df, first_pass, final_recs = simulate_time_rollout(
            dr_by_segment=results["dr_by_segment"],
            baseline=results["meta"]["baseline"],
            margin=results["meta"]["margin"],
            episodes_per_day=20,
            max_days=14,
            start_segments=("Medium", "High"),
            seed=0,
        )
        print("\n=== TIME-BASED ROLLOUT SIM (quick) ===")
        print("First pass day:", first_pass)
        print("Final recs:", final_recs)
        # Uncomment to print last day rows
        # print(timeline_df[timeline_df["day"] == timeline_df["day"].max()])
    except Exception as e:
        print("\n(Time rollout sim skipped in CLI):", str(e))


if __name__ == "__main__":
    main()
