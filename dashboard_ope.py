# dashboard_ope.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analysis_ope import run_ope


# =============================
# Global plot style (compact)
# =============================
PLOT_KW = dict(figsize=(4.0, 2.5), dpi=50)


# =============================
# Plot helpers
# =============================
def ci_bar_plot(ci_dict, baseline=0.0, cql_score=None):
    segs = list(ci_dict.keys())
    est = [ci_dict[s]["estimate"] for s in segs]
    lo = [ci_dict[s]["ci_low"] for s in segs]
    hi = [ci_dict[s]["ci_high"] for s in segs]
    yerr = [
        [e - l for e, l in zip(est, lo)],
        [h - e for e, h in zip(est, hi)],
    ]

    fig = plt.figure(**PLOT_KW)
    plt.errorbar(
        segs, est, yerr=yerr,
        fmt="o", capsize=4, linewidth=1.2, markersize=6
    )
    plt.axhline(baseline, linestyle="--", linewidth=1.2, label="Baseline")
    if cql_score is not None:
        plt.axhline(cql_score, linestyle="--", linewidth=1.2, label="CQL")

    plt.title("Segment-wise DR (95% CI)")
    plt.ylabel("Estimated Return")
    plt.xlabel("Segment")
    plt.legend()
    plt.tight_layout()
    return fig


def dist_plot(overall_samples):
    fig = plt.figure(figsize=(4.0, 2.5), dpi=50)
    for name, samples in overall_samples.items():
        plt.hist(samples, bins=25, alpha=0.5, label=name)

    plt.title("OPE Distribution (Episode-level)")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    return fig


# =============================
# Page config
# =============================
st.set_page_config(
    page_title="OPE Dashboard",
    layout="wide",
    page_icon="📊"
)


# =============================
# Sidebar navigation
# =============================
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Business Simulator", "OPE Dashboard"],
        index=0,
    )
    st.markdown("---")


# ============================================================
# ============================================================
# PAGE 1 — BUSINESS SIMULATOR
# ============================================================
# ============================================================

SCENARIOS = {
    "E-commerce Dynamic Pricing": {
        "icon": "🛒",
        "company_type": "Online Retailer (Amazon-style)",
        "rl_controls": "Product pricing across thousands of SKUs in real-time",
        "deploy_means": "Replacing static price rules with an ML-driven pricing engine",
        "base_revenue": 15_000_000,
        "baseline_return": -50.0,
        "new_return": 80.0,
        "improvement_pct": 0.15,
        "segments": ["Budget Shoppers", "Mainstream", "Premium"],
        "seg_returns": [30.0, 75.0, 110.0],
        "seg_baselines": [-20.0, -50.0, -80.0],
        "color": "#FF9900",
    },
    "Recommendation Engine": {
        "icon": "🎬",
        "company_type": "Streaming Platform (Netflix / Spotify-style)",
        "rl_controls": "Content recommendations per user session to maximise watch-time",
        "deploy_means": "Upgrading the recommendation algorithm from collaborative filtering to RL",
        "base_revenue": 8_000_000,
        "baseline_return": -30.0,
        "new_return": 55.0,
        "improvement_pct": 0.12,
        "segments": ["Casual Viewers", "Regular", "Power Users"],
        "seg_returns": [20.0, 50.0, 90.0],
        "seg_baselines": [-15.0, -30.0, -45.0],
        "color": "#E50914",
    },
    "Customer Incentives": {
        "icon": "🎁",
        "company_type": "Retail / Loyalty Platform",
        "rl_controls": "Discount and coupon targeting per customer segment",
        "deploy_means": "Replacing blanket discounts with personalised incentive allocation",
        "base_revenue": 5_000_000,
        "baseline_return": 0.0,
        "new_return": 95.0,
        "improvement_pct": 0.18,
        "segments": ["Low Value", "Medium Value", "High Value"],
        "seg_returns": [40.0, 90.0, 140.0],
        "seg_baselines": [0.0, 0.0, 0.0],
        "color": "#00A651",
    },
}


def _simulate_ope(scenario_key: str, budget: int, risk_tolerance: str, days: int, seed: int = 42):
    """
    Generate synthetic but realistic OPE outputs for the Business Simulator.
    Maps scenario + slider inputs to DR estimates and confidence intervals.
    """
    sc = SCENARIOS[scenario_key]
    rng = np.random.default_rng(seed)

    noise_scale = {"Conservative": 0.05, "Balanced": 0.12, "Aggressive": 0.22}[risk_tolerance]
    n_episodes = max(50, int(budget / 2000))

    baseline = sc["baseline_return"]
    new_ret = sc["new_return"]

    # Simulate per-episode DR values for the new policy
    dr_samples = rng.normal(loc=new_ret, scale=abs(new_ret) * noise_scale + 10, size=n_episodes)
    dr_mean = float(dr_samples.mean())

    # Bootstrap CI
    boots = [rng.choice(dr_samples, size=n_episodes, replace=True).mean() for _ in range(500)]
    ci_low = float(np.percentile(boots, 2.5))
    ci_high = float(np.percentile(boots, 97.5))
    ci_width = ci_high - ci_low

    # Segment-level DR
    seg_ci = {}
    for i, seg in enumerate(sc["segments"]):
        seg_mean = sc["seg_returns"][i]
        seg_base = sc["seg_baselines"][i]
        seg_samples = rng.normal(loc=seg_mean, scale=abs(seg_mean) * noise_scale + 8, size=max(20, n_episodes // 3))
        s_boots = [rng.choice(seg_samples, size=len(seg_samples), replace=True).mean() for _ in range(300)]
        seg_ci[seg] = {
            "estimate": float(seg_samples.mean()),
            "ci_low": float(np.percentile(s_boots, 2.5)),
            "ci_high": float(np.percentile(s_boots, 97.5)),
            "baseline": seg_base,
        }

    # Business metric derivations
    improvement_pct = max(0.0, (dr_mean - baseline) / (abs(baseline) + 1e-8))
    revenue_impact = sc["base_revenue"] * sc["improvement_pct"] * (days / 90) * (budget / 500_000)
    confidence_level = max(50.0, min(99.0, 100.0 * (1.0 - ci_width / (abs(dr_mean) * 2 + 1e-8))))

    gate_pass = ci_low >= baseline
    segs_passing = sum(1 for v in seg_ci.values() if v["ci_low"] >= v["baseline"])

    risk_label, risk_color = _risk_label(ci_low, baseline, noise_scale)

    return {
        "dr_mean": dr_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
        "baseline": baseline,
        "revenue_impact": revenue_impact,
        "confidence_level": confidence_level,
        "gate_pass": gate_pass,
        "risk_label": risk_label,
        "risk_color": risk_color,
        "improvement_pct": improvement_pct,
        "seg_ci": seg_ci,
        "segs_passing": segs_passing,
        "n_segs": len(sc["segments"]),
        "days": days,
        "scenario": scenario_key,
    }


def _risk_label(ci_low, baseline, noise_scale):
    margin = ci_low - baseline
    if margin > 20 and noise_scale <= 0.12:
        return "LOW", "#00A651"
    elif margin > 0:
        return "MEDIUM", "#FF9900"
    else:
        return "HIGH", "#E50914"


def _plain_english(sim: dict, scenario_key: str) -> str:
    sc = SCENARIOS[scenario_key]
    decision = "RECOMMENDED" if sim["gate_pass"] else "NOT RECOMMENDED"
    risk = sim["risk_label"]
    revenue = sim["revenue_impact"]
    conf = sim["confidence_level"]
    improvement = sim["improvement_pct"] * 100
    days = sim["days"]
    segs_pass = sim["segs_passing"]
    n_segs = sim["n_segs"]

    revenue_fmt = f"${revenue:,.0f}" if revenue < 1_000_000 else f"${revenue / 1_000_000:.1f}M"

    return (
        f"Based on the Off-Policy Evaluation (OPE) analysis, deploying the new "
        f"{scenario_key.lower()} policy is **{decision}**. "
        f"The model estimates a **{improvement:.0f}% performance improvement** with "
        f"**{conf:.0f}% confidence**, representing approximately **{revenue_fmt} in "
        f"additional revenue** over {days} days. "
        f"Risk is **{risk}** — the policy outperforms the baseline in "
        f"**{segs_pass} of {n_segs}** customer segments."
    )


def _export_summary(sim: dict, scenario_key: str, budget: int, risk_tolerance: str) -> str:
    sc = SCENARIOS[scenario_key]
    revenue_fmt = (
        f"${sim['revenue_impact']:,.0f}"
        if sim["revenue_impact"] < 1_000_000
        else f"${sim['revenue_impact'] / 1_000_000:.2f}M"
    )
    lines = [
        "=" * 60,
        "EXECUTIVE SUMMARY — OPE BUSINESS SIMULATION",
        "=" * 60,
        f"Scenario        : {scenario_key}",
        f"Company Type    : {sc['company_type']}",
        f"Policy Controls : {sc['rl_controls']}",
        f"Deployment      : {sc['deploy_means']}",
        "",
        "SIMULATION PARAMETERS",
        f"  Budget at Risk : ${budget:,}",
        f"  Risk Tolerance : {risk_tolerance}",
        f"  Time Horizon   : {sim['days']} days",
        "",
        "KEY OUTCOMES",
        f"  Revenue Impact     : {revenue_fmt}",
        f"  Risk Score         : {sim['risk_label']}",
        f"  Deployment Decision: {'DEPLOY' if sim['gate_pass'] else 'HOLD'}",
        f"  Confidence Level   : {sim['confidence_level']:.1f}%",
        "",
        "OPE TECHNICAL METRICS",
        f"  DR Estimate : {sim['dr_mean']:.2f}",
        f"  95% CI      : [{sim['ci_low']:.2f}, {sim['ci_high']:.2f}]",
        f"  Baseline    : {sim['baseline']:.2f}",
        f"  Improvement : {sim['improvement_pct'] * 100:.1f}%",
        "",
        "SEGMENT BREAKDOWN",
    ]
    for seg, v in sim["seg_ci"].items():
        gate = "PASS" if v["ci_low"] >= v["baseline"] else "HOLD"
        lines.append(
            f"  {seg:<20} DR={v['estimate']:.1f}  CI=[{v['ci_low']:.1f}, {v['ci_high']:.1f}]  {gate}"
        )
    lines += [
        "",
        "PLAIN ENGLISH SUMMARY",
        _plain_english(sim, scenario_key).replace("**", ""),
        "",
        "=" * 60,
        "Generated by OPE Dashboard — Business Simulator",
        "=" * 60,
    ]
    return "\n".join(lines)


def render_business_simulator():
    st.title("Business Simulator")
    st.caption(
        "Translate Off-Policy Evaluation results into business outcomes — "
        "designed for non-technical stakeholders."
    )
    st.markdown("---")

    # ── Scenario cards ──────────────────────────────────────────────
    st.subheader("Select a Scenario")

    col1, col2, col3 = st.columns(3)
    scenario_cols = [col1, col2, col3]
    scenario_keys = list(SCENARIOS.keys())

    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = scenario_keys[2]  # Customer Incentives by default

    for col, key in zip(scenario_cols, scenario_keys):
        sc = SCENARIOS[key]
        selected = st.session_state.selected_scenario == key
        border_style = f"3px solid {sc['color']}" if selected else "1px solid #444"
        bg_style = "#1a1a2e" if selected else "#0e0e0e"
        with col:
            st.markdown(
                f"""
                <div style="border:{border_style}; border-radius:10px; padding:16px;
                            background:{bg_style}; min-height:160px;">
                  <div style="font-size:2rem">{sc['icon']}</div>
                  <div style="font-weight:700; font-size:1rem; margin-top:4px">{key}</div>
                  <div style="font-size:0.78rem; color:#aaa; margin-top:6px">
                    <b>Company:</b> {sc['company_type']}<br>
                    <b>RL controls:</b> {sc['rl_controls']}<br>
                    <b>Deploy means:</b> {sc['deploy_means']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                f"Select {'✓' if selected else ''}",
                key=f"sel_{key}",
                use_container_width=True,
            ):
                st.session_state.selected_scenario = key
                st.rerun()

    st.markdown("---")
    selected_scenario = st.session_state.selected_scenario
    sc = SCENARIOS[selected_scenario]

    # ── Sidebar simulation controls ──────────────────────────────────
    with st.sidebar:
        st.header("Simulation Controls")
        sidebar_scenario = st.selectbox(
            "Scenario",
            scenario_keys,
            index=scenario_keys.index(selected_scenario),
            key="sidebar_scenario",
        )
        if sidebar_scenario != selected_scenario:
            st.session_state.selected_scenario = sidebar_scenario
            st.rerun()

        budget = st.slider(
            "Budget at Risk ($)",
            min_value=10_000,
            max_value=1_000_000,
            value=250_000,
            step=10_000,
            format="$%d",
        )
        risk_tolerance = st.radio(
            "Risk Tolerance",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
        )
        time_horizon = st.select_slider(
            "Time Horizon (days)",
            options=[30, 60, 90],
            value=90,
        )
        st.markdown("---")

    # ── Run simulation ───────────────────────────────────────────────
    sim = _simulate_ope(selected_scenario, budget, risk_tolerance, time_horizon)

    # ── Business Outcome Cards ────────────────────────────────────────
    st.subheader(f"{sc['icon']}  {selected_scenario} — Business Outcomes")

    revenue_fmt = (
        f"${sim['revenue_impact']:,.0f}"
        if sim['revenue_impact'] < 1_000_000
        else f"+${sim['revenue_impact'] / 1_000_000:.1f}M"
    )
    deploy_label = "DEPLOY" if sim["gate_pass"] else "HOLD"
    deploy_color = "#00A651" if sim["gate_pass"] else "#E50914"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div style="background:#1e1e2e; border-radius:10px; padding:20px; text-align:center;
                        border-left:5px solid #00A651;">
              <div style="font-size:0.85rem; color:#aaa; text-transform:uppercase;">Revenue Impact</div>
              <div style="font-size:2.2rem; font-weight:800; color:#00A651; margin-top:6px">
                {revenue_fmt}
              </div>
              <div style="font-size:0.75rem; color:#666; margin-top:4px">over {time_horizon} days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        risk_color = sim["risk_color"]
        st.markdown(
            f"""
            <div style="background:#1e1e2e; border-radius:10px; padding:20px; text-align:center;
                        border-left:5px solid {risk_color};">
              <div style="font-size:0.85rem; color:#aaa; text-transform:uppercase;">Risk Score</div>
              <div style="font-size:2.2rem; font-weight:800; color:{risk_color}; margin-top:6px">
                {sim['risk_label']}
              </div>
              <div style="font-size:0.75rem; color:#666; margin-top:4px">
                DR CI low: {sim['ci_low']:.1f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div style="background:#1e1e2e; border-radius:10px; padding:20px; text-align:center;
                        border-left:5px solid {deploy_color};">
              <div style="font-size:0.85rem; color:#aaa; text-transform:uppercase;">Deployment Decision</div>
              <div style="font-size:2.2rem; font-weight:800; color:{deploy_color}; margin-top:6px">
                {deploy_label}
              </div>
              <div style="font-size:0.75rem; color:#666; margin-top:4px">policy gate result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div style="background:#1e1e2e; border-radius:10px; padding:20px; text-align:center;
                        border-left:5px solid #4A90E2;">
              <div style="font-size:0.85rem; color:#aaa; text-transform:uppercase;">Confidence Level</div>
              <div style="font-size:2.2rem; font-weight:800; color:#4A90E2; margin-top:6px">
                {sim['confidence_level']:.0f}%
              </div>
              <div style="font-size:0.75rem; color:#666; margin-top:4px">
                CI width: {sim['ci_width']:.1f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Before vs After Comparison ────────────────────────────────────
    st.subheader("Before vs After — Policy Comparison")

    left_col, arrow_col, right_col = st.columns([5, 1, 5])

    baseline_val = sc["baseline_return"]
    new_val = sim["dr_mean"]
    delta_pct = sim["improvement_pct"] * 100
    delta_rev = sim["revenue_impact"]
    delta_rev_fmt = (
        f"${delta_rev:,.0f}" if delta_rev < 1_000_000 else f"${delta_rev / 1_000_000:.1f}M"
    )

    with left_col:
        st.markdown(
            f"""
            <div style="background:#1a0a0a; border:1px solid #E50914; border-radius:10px; padding:20px;">
              <div style="font-size:1rem; font-weight:700; color:#E50914; margin-bottom:12px">
                Current Policy (Baseline)
              </div>
              <table style="width:100%; font-size:0.9rem; color:#ccc; border-collapse:collapse;">
                <tr>
                  <td style="padding:6px 0; color:#aaa">DR Estimate</td>
                  <td style="padding:6px 0; font-weight:600">{baseline_val:.1f}</td>
                </tr>
                <tr>
                  <td style="padding:6px 0; color:#aaa">Revenue Impact</td>
                  <td style="padding:6px 0; font-weight:600">$0 (reference)</td>
                </tr>
                <tr>
                  <td style="padding:6px 0; color:#aaa">Risk</td>
                  <td style="padding:6px 0; font-weight:600">—</td>
                </tr>
                <tr>
                  <td style="padding:6px 0; color:#aaa">Confidence</td>
                  <td style="padding:6px 0; font-weight:600">—</td>
                </tr>
                <tr>
                  <td style="padding:6px 0; color:#aaa">Deployment</td>
                  <td style="padding:6px 0; font-weight:600; color:#aaa">Active (current)</td>
                </tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arrow_col:
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:center;
                        height:100%; font-size:2rem; color:#4A90E2; padding-top:50px;">
              ➜
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            f"""
            <div style="background:#0a1a0a; border:1px solid #00A651; border-radius:10px; padding:20px;">
              <div style="font-size:1rem; font-weight:700; color:#00A651; margin-bottom:12px">
                New Policy (Target)
              </div>
              <table style="width:100%; font-size:0.9rem; color:#ccc; border-collapse:collapse;">
                <tr>
                  <td style="padding:6px 0; color:#aaa">DR Estimate</td>
                  <td style="padding:6px 0; font-weight:600;color:#00A651">
                    {new_val:.1f}
                    <span style="font-size:0.75rem; color:#aaa">
                      (+{new_val - baseline_val:.1f})
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style="padding:6px 0; color:#aaa">Revenue Impact</td>
                  <td style="padding:6px 0; font-weight:600; color:#00A651">
                    +{delta_rev_fmt}
                    <span style="font-size:0.75rem; color:#aaa">(+{delta_pct:.0f}%)</span>
                  </td>
                </tr>
                <tr>
                  <td style="padding:6px 0; color:#aaa">Risk</td>
                  <td style="padding:6px 0; font-weight:600;
                             color:{sim['risk_color']}">{sim['risk_label']}</td>
                </tr>
                <tr>
                  <td style="padding:6px 0; color:#aaa">Confidence</td>
                  <td style="padding:6px 0; font-weight:600; color:#4A90E2">
                    {sim['confidence_level']:.0f}%
                  </td>
                </tr>
                <tr>
                  <td style="padding:6px 0; color:#aaa">Deployment</td>
                  <td style="padding:6px 0; font-weight:600;
                             color:{deploy_color}">{deploy_label}</td>
                </tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Segment breakdown chart ───────────────────────────────────────
    st.subheader("Segment-level DR — Confidence Intervals")

    seg_names = list(sim["seg_ci"].keys())
    seg_est = [sim["seg_ci"][s]["estimate"] for s in seg_names]
    seg_lo = [sim["seg_ci"][s]["ci_low"] for s in seg_names]
    seg_hi = [sim["seg_ci"][s]["ci_high"] for s in seg_names]
    seg_bases = [sim["seg_ci"][s]["baseline"] for s in seg_names]

    yerr = [
        [e - l for e, l in zip(seg_est, seg_lo)],
        [h - e for e, h in zip(seg_est, seg_hi)],
    ]

    fig, ax = plt.subplots(figsize=(6, 3), dpi=60)
    colors_seg = [
        sc["color"] if v["ci_low"] >= v["baseline"] else "#E50914"
        for v in sim["seg_ci"].values()
    ]
    ax.errorbar(
        seg_names, seg_est, yerr=yerr,
        fmt="o", capsize=5, linewidth=1.5, markersize=8,
        color="#4A90E2",
    )
    for i, (seg, base) in enumerate(zip(seg_names, seg_bases)):
        ax.axhline(base, linestyle="--", linewidth=1.0, alpha=0.5, color="#aaa")
    ax.set_title("Segment DR vs Baseline (95% CI)", fontsize=10)
    ax.set_ylabel("Estimated Return")
    ax.set_xlabel("Customer Segment")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=False)

    st.markdown("---")

    # ── Plain English Explanation ─────────────────────────────────────
    st.subheader("Plain English Summary")

    summary_text = _plain_english(sim, selected_scenario)
    box_color = "#0a1a0a" if sim["gate_pass"] else "#1a0a0a"
    border_color = "#00A651" if sim["gate_pass"] else "#E50914"
    st.markdown(
        f"""
        <div style="background:{box_color}; border-left:5px solid {border_color};
                    border-radius:6px; padding:20px; font-size:1.05rem;
                    line-height:1.7; color:#e0e0e0;">
          {summary_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Executive Summary Export ──────────────────────────────────────
    st.subheader("Export Executive Summary")

    export_text = _export_summary(sim, selected_scenario, budget, risk_tolerance)
    buf = io.BytesIO(export_text.encode("utf-8"))

    col_dl, col_preview = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="Download Summary (.txt)",
            data=buf,
            file_name=f"ope_summary_{selected_scenario.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_preview:
        with st.expander("Preview export content"):
            st.code(export_text, language="text")


# ============================================================
# ============================================================
# PAGE 2 — OPE DASHBOARD (existing)
# ============================================================
# ============================================================

def render_ope_dashboard():
    st.title("📊 Off-Policy Evaluation (OPE) Dashboard")
    st.caption("IPS · WIS · DM · DR · Bootstrap CI · Policy Gate")

    st.markdown("---")

    with st.sidebar:
        st.header("📥 Inputs")
        log_path = st.text_input("Logged data (.npz)", "data/logged_behavior.npz")
        model_path = st.text_input("Target policy (.pth)", "dqn_policy.pth")

        st.header("⚙️ OPE Settings")
        n_boot = st.slider("Bootstrap samples", 200, 2000, 800, step=100)
        clip_rho = st.slider("Importance weight clip (ρ)", 5.0, 200.0, 50.0, step=5.0)

        st.header("🚦 Policy Gate")
        baseline = st.number_input("Baseline return", value=0.0, step=10.0)
        margin = st.number_input("Safety margin", value=0.0, step=10.0)

        st.header("🧪 CQL Overlay")
        use_cql = st.checkbox("Compare against CQL", value=False)
        cql_score = st.number_input("CQL score", value=0.0, step=10.0) if use_cql else None

        run_btn = st.button("🚀 Run OPE")

    @st.cache_data(show_spinner=False)
    def cached_run(log_path, model_path, n_boot, baseline, margin, clip_rho):
        return run_ope(
            log_path=log_path,
            model_path=model_path,
            n_boot=n_boot,
            baseline=baseline,
            margin=margin,
            clip_rho=clip_rho,
            quiet=True,
        )

    if run_btn:
        with st.spinner("Running off-policy evaluation…"):
            results = cached_run(log_path, model_path, n_boot, baseline, margin, clip_rho)

        overall = results["overall"]
        seg_ci = results["segment"]
        rollout = results["rollout"]

        dr = overall["DR"]
        deploy_pass = dr["ci_low"] >= baseline + margin

        if deploy_pass:
            st.success("✅ **Deployment decision: PASS** — DR lower bound clears safety gate.")
        else:
            st.error("⛔ **Deployment decision: HOLD** — DR lower bound below safety gate.")

        st.markdown("---")

        st.subheader("📌 Overall OPE Estimates")

        cols = st.columns(4)
        for col, key in zip(cols, ["IPS", "WIS", "DM", "DR"]):
            v = overall[key]
            col.metric(
                label=key,
                value=f"{v['estimate']:.2f}",
                delta=f"[{v['ci_low']:.2f}, {v['ci_high']:.2f}]"
            )

        st.caption(
            f"Episodes={results['meta']['episodes']} · "
            f"ρ_clip={results['meta']['clip_rho']} · "
            f"baseline={baseline} · margin={margin}"
        )

        st.markdown("---")

        st.subheader("📊 Segment-wise Confidence Intervals")
        st.pyplot(
            ci_bar_plot(seg_ci, baseline, cql_score),
            clear_figure=True,
            use_container_width=False
        )

        st.markdown("---")

        st.subheader("🧭 Partial Rollout Recommendations")

        rows = []
        for seg in seg_ci:
            rows.append({
                "Segment": seg,
                "DR Mean": seg_ci[seg]["estimate"],
                "CI Low": seg_ci[seg]["ci_low"],
                "CI High": seg_ci[seg]["ci_high"],
                "Gate": "PASS" if seg_ci[seg]["ci_low"] >= baseline + margin else "HOLD",
                "Suggested Rollout": rollout[seg]["rollout"],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.info("Gate rule: **PASS if CI_low ≥ baseline + margin**")

        st.markdown("---")

        st.subheader("📈 OPE Sample Distributions")
        st.pyplot(
            dist_plot({
                "IPS": overall["IPS"]["samples"],
                "WIS": overall["WIS"]["samples"],
                "DM": overall["DM"]["samples"],
                "DR": overall["DR"]["samples"],
            }),
            clear_figure=True,
            use_container_width=False
        )

        st.markdown("---")

        st.subheader("🧪 CQL vs OPE DR")

        if use_cql:
            st.write(
                f"**CQL score:** {cql_score:.2f}\n\n"
                f"**OPE DR:** {dr['estimate']:.2f} "
                f"(CI [{dr['ci_low']:.2f}, {dr['ci_high']:.2f}])"
            )

            if dr["ci_low"] > cql_score:
                st.success("DR lower bound exceeds CQL → policy stronger than conservative offline baseline.")
            else:
                st.warning("DR lower bound below CQL → conservative baseline still safer.")
        else:
            st.info("Enable CQL overlay to compare conservative offline baseline.")

    else:
        st.info("⬅️ Configure inputs and click **Run OPE**.")


# ============================================================
# Router
# ============================================================
if page == "Business Simulator":
    render_business_simulator()
else:
    render_ope_dashboard()
