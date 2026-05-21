# dashboard_ope.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
    lo  = [ci_dict[s]["ci_low"]  for s in segs]
    hi  = [ci_dict[s]["ci_high"] for s in segs]
    yerr = [
        [e - l for e, l in zip(est, lo)],
        [h - e for e, h in zip(est, hi)],
    ]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(**PLOT_KW)
    fig.patch.set_facecolor('#080812')
    ax.set_facecolor('#0d0d1a')

    ax.errorbar(segs, est, yerr=yerr, fmt="o", capsize=4, linewidth=1.2,
                markersize=6, color='#06B6D4', ecolor='#06B6D4')
    ax.axhline(baseline, linestyle="--", linewidth=1.2, color='#EF4444', label="Baseline")
    if cql_score is not None:
        ax.axhline(cql_score, linestyle="--", linewidth=1.2, color='#7C3AED', label="CQL")

    ax.set_title("Segment-wise DR (95% CI)", color='white')
    ax.set_ylabel("Estimated Return", color='white')
    ax.set_xlabel("Segment", color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ffffff26')
    ax.grid(color='#ffffff', alpha=0.06, linewidth=0.6)

    legend = ax.legend(facecolor='#0d0d1a', edgecolor='#ffffff26', labelcolor='white')
    plt.tight_layout()
    return fig


def dist_plot(overall_samples):
    _HIST_COLORS = ['#7C3AED', '#06B6D4', '#10B981', '#3B82F6']

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4.0, 2.5), dpi=50)
    fig.patch.set_facecolor('#080812')
    ax.set_facecolor('#0d0d1a')

    for (name, samples), color in zip(overall_samples.items(), _HIST_COLORS):
        ax.hist(samples, bins=25, alpha=0.55, label=name, color=color)

    ax.set_title("OPE Distribution (Episode-level)", color='white')
    ax.set_xlabel("Return", color='white')
    ax.set_ylabel("Count", color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ffffff26')
    ax.grid(color='#ffffff', alpha=0.06, linewidth=0.6)

    ax.legend(facecolor='#0d0d1a', edgecolor='#ffffff26', labelcolor='white')
    plt.tight_layout()
    return fig


# =============================
# Page config  (must be first)
# =============================
st.set_page_config(page_title="OPE Dashboard", layout="wide", page_icon="📊")

# ── theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── global ───────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #F1F5F9 !important;
}

.stApp {
    background: #080812 !important;
}

/* ── sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}
[data-testid="stSidebar"] .stMarkdown hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* sidebar section headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a78bfa !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 600 !important;
}

/* ── radio buttons ────────────────────────────────────────────────── */
[data-testid="stRadio"] label {
    color: #94A3B8 !important;
    font-weight: 500 !important;
    transition: color .15s !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    color: #a78bfa !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: inherit !important;
}
/* radio circle accent */
[data-testid="stRadio"] input[type="radio"]:checked + div {
    background: #7C3AED !important;
    border-color: #7C3AED !important;
}

/* ── sliders ──────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #7C3AED !important;
    border-color: #7C3AED !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid] {
    background: linear-gradient(90deg, #7C3AED, #3B82F6) !important;
}

/* ── buttons ──────────────────────────────────────────────────────── */
[data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"],
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #3B82F6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity .2s, transform .2s, box-shadow .2s !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(124,58,237,0.5) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* download button */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #7C3AED, #3B82F6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.3) !important;
}

/* ── inputs & text areas ──────────────────────────────────────────── */
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-baseweb="input"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important;
}

/* selectbox */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
}
[data-baseweb="select"] svg { color: #94A3B8 !important; }
[data-baseweb="popover"] [role="listbox"] {
    background: #13132b !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
}
[data-baseweb="popover"] [role="option"] {
    background: transparent !important;
    color: #F1F5F9 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: rgba(124,58,237,0.2) !important;
    color: #a78bfa !important;
}

/* ── checkboxes ───────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label { color: #94A3B8 !important; }
[data-testid="stCheckbox"] input:checked + div {
    background: #7C3AED !important;
    border-color: #7C3AED !important;
}

/* ── metric cards ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-left: 3px solid #7C3AED !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetric"] label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #06B6D4 !important;
}

/* ── dataframe / tables ───────────────────────────────────────────── */
[data-testid="stDataFrame"] iframe,
.stDataFrame {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}

/* ── expanders ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #94A3B8 !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { color: #F1F5F9 !important; }

/* ── info / success / warning / error banners ─────────────────────── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}
/* info */
div[data-testid="stAlert"].st-ae {
    background: rgba(59,130,246,0.10) !important;
    border-color: #3B82F6 !important;
    color: #93c5fd !important;
}
/* success */
.stSuccess, [data-testid="stAlert"][kind="success"] {
    background: rgba(16,185,129,0.10) !important;
    border-color: #10B981 !important;
    color: #6ee7b7 !important;
}
.stSuccess * { color: #6ee7b7 !important; }
/* error / hold */
.stError, [data-testid="stAlert"][kind="error"] {
    background: rgba(239,68,68,0.10) !important;
    border-color: #EF4444 !important;
    color: #fca5a5 !important;
}
.stError * { color: #fca5a5 !important; }
/* warning */
.stWarning, [data-testid="stAlert"][kind="warning"] {
    background: rgba(245,158,11,0.10) !important;
    border-color: #F59E0B !important;
    color: #fde68a !important;
}

/* ── page headings ────────────────────────────────────────────────── */
h1 { color: #F1F5F9 !important; font-weight: 800 !important; }
h2 { color: #F1F5F9 !important; font-weight: 700 !important; }
h3 {
    color: #a78bfa !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── dividers ─────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── code blocks ──────────────────────────────────────────────────── */
code, pre, [data-testid="stCode"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #a78bfa !important;
}

/* ── caption / small text ─────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: #475569 !important;
    font-size: 0.8rem !important;
}

/* ── spinner ──────────────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: #7C3AED !important; }

/* ── main content padding ─────────────────────────────────────────── */
.block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}
</style>
""", unsafe_allow_html=True)

# ── top-level navigation ──────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Business Simulator", "OPE Dashboard"])
st.sidebar.markdown("---")


# ==============================================================================
# BUSINESS SIMULATOR
# ==============================================================================

# ── scenario definitions ──────────────────────────────────────────────────────
_SCENARIOS = {
    "E-commerce Dynamic Pricing": {
        "icon": "🛒",
        "company_type": "Online Retailer (Amazon-style)",
        "rl_controls": "Product pricing across thousands of SKUs in real-time",
        "deploy_means": "Replace static price rules with an ML-driven pricing engine",
        "base_revenue": 15_000_000,
        "new_return": 80.0,
        "baseline_return": -50.0,
        "improvement_pct": 0.15,
        "segments": ["Budget Shoppers", "Mainstream", "Premium"],
        "seg_new":      [30.0, 75.0, 110.0],
        "seg_baseline": [-20.0, -50.0, -80.0],
        "accent": "#FF9900",
    },
    "Recommendation Engine": {
        "icon": "🎬",
        "company_type": "Streaming Platform (Netflix / Spotify-style)",
        "rl_controls": "Content recommendations per session to maximise watch-time",
        "deploy_means": "Upgrade collaborative filtering to an RL recommendation policy",
        "base_revenue": 8_000_000,
        "new_return": 55.0,
        "baseline_return": -30.0,
        "improvement_pct": 0.12,
        "segments": ["Casual Viewers", "Regular", "Power Users"],
        "seg_new":      [20.0, 50.0, 90.0],
        "seg_baseline": [-15.0, -30.0, -45.0],
        "accent": "#E50914",
    },
    "Customer Incentives": {
        "icon": "🎁",
        "company_type": "Retail / Loyalty Platform",
        "rl_controls": "Discount and coupon targeting per customer segment",
        "deploy_means": "Replace blanket discounts with personalised incentive allocation",
        "base_revenue": 5_000_000,
        "new_return": 95.0,
        "baseline_return": 0.0,
        "improvement_pct": 0.18,
        "segments": ["Low Value", "Medium Value", "High Value"],
        "seg_new":      [40.0, 90.0, 140.0],
        "seg_baseline": [0.0, 0.0, 0.0],
        "accent": "#00A651",
    },
}


def _run_sim(scenario_key: str, budget: int, risk_tolerance: str, days: int):
    """Generate synthetic but realistic OPE outputs for the Business Simulator."""
    sc  = _SCENARIOS[scenario_key]
    rng = np.random.default_rng(42)

    noise = {"Conservative": 0.05, "Balanced": 0.12, "Aggressive": 0.22}[risk_tolerance]
    n = max(60, budget // 2_000)

    # overall DR for the new policy
    dr_samples = rng.normal(loc=sc["new_return"],
                            scale=abs(sc["new_return"]) * noise + 10,
                            size=n)
    dr_mean = float(dr_samples.mean())
    boots   = [rng.choice(dr_samples, size=n, replace=True).mean() for _ in range(600)]
    ci_low  = float(np.percentile(boots, 2.5))
    ci_high = float(np.percentile(boots, 97.5))

    # per-segment DR
    seg_ci = {}
    for i, seg in enumerate(sc["segments"]):
        s_samp  = rng.normal(loc=sc["seg_new"][i],
                             scale=abs(sc["seg_new"][i]) * noise + 8,
                             size=max(20, n // 3))
        s_boots = [rng.choice(s_samp, size=len(s_samp), replace=True).mean()
                   for _ in range(300)]
        seg_ci[seg] = {
            "estimate": float(s_samp.mean()),
            "ci_low":   float(np.percentile(s_boots, 2.5)),
            "ci_high":  float(np.percentile(s_boots, 97.5)),
            "baseline": sc["seg_baseline"][i],
        }

    gate_pass      = ci_low >= sc["baseline_return"]
    ci_width       = ci_high - ci_low
    improvement    = max(0.0, (dr_mean - sc["baseline_return"]) /
                         (abs(sc["baseline_return"]) + 1e-8))
    revenue_impact = sc["base_revenue"] * sc["improvement_pct"] * (days / 90) * (budget / 500_000)
    confidence     = max(50.0, min(99.0,
                         100.0 * (1.0 - ci_width / (abs(dr_mean) * 2 + 1e-8))))
    segs_pass      = sum(1 for v in seg_ci.values() if v["ci_low"] >= v["baseline"])

    if ci_low >= sc["baseline_return"] + 20 and noise <= 0.12:
        risk_label, risk_color = "LOW",    "#00A651"
    elif gate_pass:
        risk_label, risk_color = "MEDIUM", "#FF9900"
    else:
        risk_label, risk_color = "HIGH",   "#E50914"

    return {
        "dr_mean": dr_mean, "ci_low": ci_low, "ci_high": ci_high,
        "ci_width": ci_width, "baseline": sc["baseline_return"],
        "revenue_impact": revenue_impact, "confidence": confidence,
        "gate_pass": gate_pass, "risk_label": risk_label, "risk_color": risk_color,
        "improvement_pct": improvement, "seg_ci": seg_ci,
        "segs_pass": segs_pass, "n_segs": len(sc["segments"]), "days": days,
    }


def _fmt_revenue(v: float) -> str:
    return f"${v / 1_000_000:.1f}M" if v >= 1_000_000 else f"${v:,.0f}"


def _plain_english(sim: dict, sc_key: str) -> str:
    sc       = _SCENARIOS[sc_key]
    decision = "RECOMMENDED" if sim["gate_pass"] else "NOT RECOMMENDED"
    rev      = _fmt_revenue(sim["revenue_impact"])
    impr     = sim["improvement_pct"] * 100
    return (
        f"Based on the Off-Policy Evaluation (OPE) analysis, deploying the new "
        f"**{sc_key.lower()}** policy is **{decision}**. "
        f"The model estimates a **{impr:.0f}% performance improvement** with "
        f"**{sim['confidence']:.0f}% confidence**, representing approximately "
        f"**{rev} in additional revenue** over {sim['days']} days. "
        f"Risk is **{sim['risk_label']}** — the policy outperforms the baseline "
        f"in **{sim['segs_pass']} of {sim['n_segs']}** customer segments."
    )


def _export_text(sim: dict, sc_key: str, budget: int, risk_tolerance: str) -> str:
    sc = _SCENARIOS[sc_key]
    rev = _fmt_revenue(sim["revenue_impact"])
    lines = [
        "=" * 62,
        "EXECUTIVE SUMMARY — OPE BUSINESS SIMULATION",
        "=" * 62,
        f"Scenario         : {sc_key}",
        f"Company Type     : {sc['company_type']}",
        f"RL Controls      : {sc['rl_controls']}",
        f"Deployment Means : {sc['deploy_means']}",
        "",
        "SIMULATION PARAMETERS",
        f"  Budget at Risk  : ${budget:,}",
        f"  Risk Tolerance  : {risk_tolerance}",
        f"  Time Horizon    : {sim['days']} days",
        "",
        "BUSINESS OUTCOMES",
        f"  Revenue Impact      : +{rev}",
        f"  Risk Score          : {sim['risk_label']}",
        f"  Deployment Decision : {'DEPLOY' if sim['gate_pass'] else 'HOLD'}",
        f"  Confidence Level    : {sim['confidence']:.1f}%",
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
            f"  {seg:<22} DR={v['estimate']:.1f}  "
            f"CI=[{v['ci_low']:.1f}, {v['ci_high']:.1f}]  {gate}"
        )
    summary = _plain_english(sim, sc_key).replace("**", "")
    lines += ["", "PLAIN-ENGLISH SUMMARY", summary, "",
              "=" * 62, "Generated by OPE Dashboard — Business Simulator", "=" * 62]
    return "\n".join(lines)


if page == "Business Simulator":
    st.title("Business Simulator")
    st.caption(
        "Translate Off-Policy Evaluation results into business outcomes — "
        "designed for non-technical stakeholders."
    )
    st.markdown("---")

    # ── sidebar controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("Simulation Controls")
        sc_keys   = list(_SCENARIOS.keys())
        scenario  = st.selectbox("Scenario", sc_keys, index=2)
        budget    = st.slider("Budget at Risk ($)", 10_000, 1_000_000, 250_000,
                              step=10_000, format="$%d")
        risk_tol  = st.radio("Risk Tolerance",
                             ["Conservative", "Balanced", "Aggressive"], index=1)
        time_days = st.select_slider("Time Horizon (days)", [30, 60, 90], value=90)

    sc  = _SCENARIOS[scenario]
    sim = _run_sim(scenario, budget, risk_tol, time_days)

    # ── scenario selector cards ───────────────────────────────────────
    st.subheader("Scenario")
    col1, col2, col3 = st.columns(3)
    for col, key in zip([col1, col2, col3], sc_keys):
        s       = _SCENARIOS[key]
        active  = key == scenario
        border  = f"3px solid {s['accent']}" if active else "1px solid #444"
        bg      = "#1a1a2e"                  if active else "#111"
        col.markdown(
            f"""<div style="border:{border};border-radius:10px;padding:16px;
                background:{bg};min-height:150px;">
              <span style="font-size:1.8rem">{s['icon']}</span>
              <div style="font-weight:700;margin-top:4px">{key}</div>
              <div style="font-size:0.78rem;color:#aaa;margin-top:6px">
                <b>Company:</b> {s['company_type']}<br>
                <b>RL controls:</b> {s['rl_controls']}<br>
                <b>Deploying means:</b> {s['deploy_means']}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 4 hero metric cards ───────────────────────────────────────────
    st.subheader(f"{sc['icon']}  Business Outcomes — {scenario}")
    c1, c2, c3, c4 = st.columns(4)

    deploy_color = "#00A651" if sim["gate_pass"] else "#E50914"
    deploy_label = "DEPLOY"  if sim["gate_pass"] else "HOLD"

    def _hero(col, label, value, subtitle, accent):
        col.markdown(
            f"""<div style="background:#1e1e2e;border-radius:10px;padding:20px;
                text-align:center;border-left:5px solid {accent};">
              <div style="font-size:0.8rem;color:#aaa;text-transform:uppercase">
                {label}
              </div>
              <div style="font-size:2rem;font-weight:800;color:{accent};margin-top:6px">
                {value}
              </div>
              <div style="font-size:0.75rem;color:#666;margin-top:4px">{subtitle}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    _hero(c1, "Revenue Impact",
          f"+{_fmt_revenue(sim['revenue_impact'])}",
          f"over {time_days} days", "#00A651")
    _hero(c2, "Risk Score",
          sim["risk_label"],
          f"DR CI low: {sim['ci_low']:.1f}", sim["risk_color"])
    _hero(c3, "Deployment Decision",
          deploy_label,
          "policy gate result", deploy_color)
    _hero(c4, "Confidence Level",
          f"{sim['confidence']:.0f}%",
          f"CI width: {sim['ci_width']:.1f}", "#4A90E2")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── before vs after comparison ────────────────────────────────────
    st.subheader("Before vs After — Policy Comparison")

    base_val  = sc["baseline_return"]
    new_val   = sim["dr_mean"]
    delta_dr  = new_val - base_val
    rev_str   = f"+{_fmt_revenue(sim['revenue_impact'])}"

    left, arrow, right = st.columns([5, 1, 5])

    left.markdown(
        f"""<div style="background:#1a0a0a;border:1px solid #E50914;
            border-radius:10px;padding:20px;">
          <div style="color:#E50914;font-weight:700;margin-bottom:12px">
            Current Policy (Baseline)
          </div>
          <table style="width:100%;font-size:0.9rem;color:#ccc;border-collapse:collapse">
            <tr><td style="padding:5px 0;color:#aaa">DR Estimate</td>
                <td style="padding:5px 0;font-weight:600">{base_val:.1f}</td></tr>
            <tr><td style="padding:5px 0;color:#aaa">Revenue Impact</td>
                <td style="padding:5px 0;font-weight:600">$0 (reference)</td></tr>
            <tr><td style="padding:5px 0;color:#aaa">Risk</td>
                <td style="padding:5px 0;font-weight:600">—</td></tr>
            <tr><td style="padding:5px 0;color:#aaa">Confidence</td>
                <td style="padding:5px 0;font-weight:600">—</td></tr>
            <tr><td style="padding:5px 0;color:#aaa">Decision</td>
                <td style="padding:5px 0;font-weight:600;color:#aaa">Active (live)</td></tr>
          </table>
        </div>""",
        unsafe_allow_html=True,
    )

    arrow.markdown(
        """<div style="display:flex;align-items:center;justify-content:center;
            height:100%;font-size:2rem;color:#4A90E2;padding-top:55px">➜</div>""",
        unsafe_allow_html=True,
    )

    right.markdown(
        f"""<div style="background:#0a1a0a;border:1px solid #00A651;
            border-radius:10px;padding:20px;">
          <div style="color:#00A651;font-weight:700;margin-bottom:12px">
            New Policy (Target)
          </div>
          <table style="width:100%;font-size:0.9rem;color:#ccc;border-collapse:collapse">
            <tr><td style="padding:5px 0;color:#aaa">DR Estimate</td>
                <td style="padding:5px 0;font-weight:600;color:#00A651">
                  {new_val:.1f}
                  <span style="font-size:0.75rem;color:#aaa"> (+{delta_dr:.1f})</span>
                </td></tr>
            <tr><td style="padding:5px 0;color:#aaa">Revenue Impact</td>
                <td style="padding:5px 0;font-weight:600;color:#00A651">
                  {rev_str}
                  <span style="font-size:0.75rem;color:#aaa">
                    (+{sim['improvement_pct']*100:.0f}%)
                  </span>
                </td></tr>
            <tr><td style="padding:5px 0;color:#aaa">Risk</td>
                <td style="padding:5px 0;font-weight:600;color:{sim['risk_color']}">
                  {sim['risk_label']}
                </td></tr>
            <tr><td style="padding:5px 0;color:#aaa">Confidence</td>
                <td style="padding:5px 0;font-weight:600;color:#4A90E2">
                  {sim['confidence']:.0f}%
                </td></tr>
            <tr><td style="padding:5px 0;color:#aaa">Decision</td>
                <td style="padding:5px 0;font-weight:600;color:{deploy_color}">
                  {deploy_label}
                </td></tr>
          </table>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── segment CI chart (cyberpunk Plotly) ──────────────────────────
    st.subheader("Segment-level DR — Confidence Intervals")

    seg_names = list(sim["seg_ci"].keys())
    seg_est   = [sim["seg_ci"][s]["estimate"] for s in seg_names]
    seg_lo    = [sim["seg_ci"][s]["ci_low"]   for s in seg_names]
    seg_hi    = [sim["seg_ci"][s]["ci_high"]  for s in seg_names]
    seg_bases = [sim["seg_ci"][s]["baseline"] for s in seg_names]

    _NEON = ["#00ff41", "#00d4ff", "#bf00ff"]

    cyb_fig = go.Figure()

    for i, (seg, est, lo, hi) in enumerate(zip(seg_names, seg_est, seg_lo, seg_hi)):
        color = _NEON[i % len(_NEON)]
        err_minus = est - lo
        err_plus  = hi - est

        cyb_fig.add_trace(go.Scatter(
            x=[seg],
            y=[est],
            mode="markers",
            name=seg,
            marker=dict(
                size=14,
                color=color,
                line=dict(color=color, width=2),
            ),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[err_plus],
                arrayminus=[err_minus],
                color=color,
                thickness=1.5,
                width=6,
            ),
            showlegend=True,
        ))

    # per-segment baseline dashed lines
    for i, (seg, base) in enumerate(zip(seg_names, seg_bases)):
        color = _NEON[i % len(_NEON)]
        cyb_fig.add_shape(
            type="line",
            x0=-0.4 + i, x1=0.4 + i,
            y0=base, y1=base,
            line=dict(color="#ff4444", width=1.5, dash="dash"),
        )

    # global baseline annotation (single red dashed rule if all equal)
    if len(set(seg_bases)) == 1:
        cyb_fig.add_hline(
            y=seg_bases[0],
            line=dict(color="#ff4444", width=1.5, dash="dash"),
            annotation_text="baseline",
            annotation_font_color="#ff4444",
        )

    cyb_fig.update_layout(
        title=dict(
            text="Segment DR vs Baseline (95% CI)",
            font=dict(color="white", size=15),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(color="white"),
        xaxis=dict(
            title="Customer Segment",
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            title="Estimated Return",
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            showline=False,
        ),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        height=380,
    )

    st.plotly_chart(cyb_fig, use_container_width=True)

    st.markdown("---")

    # ── plain English summary ─────────────────────────────────────────
    st.subheader("Plain English Summary")

    summary = _plain_english(sim, scenario)
    box_bg     = "#0a1a0a" if sim["gate_pass"] else "#1a0a0a"
    box_border = "#00A651" if sim["gate_pass"] else "#E50914"
    st.markdown(
        f"""<div style="background:{box_bg};border-left:5px solid {box_border};
            border-radius:6px;padding:20px;font-size:1.05rem;
            line-height:1.75;color:#e0e0e0;">{summary}</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── executive summary download ────────────────────────────────────
    st.subheader("Export Executive Summary")

    export_txt  = _export_text(sim, scenario, budget, risk_tol)
    export_buf  = io.BytesIO(export_txt.encode("utf-8"))
    export_name = f"ope_summary_{scenario.lower().replace(' ', '_')}.txt"

    dl_col, prev_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label="Download Summary (.txt)",
            data=export_buf,
            file_name=export_name,
            mime="text/plain",
            use_container_width=True,
        )
    with prev_col:
        with st.expander("Preview export"):
            st.code(export_txt, language="text")


# ==============================================================================
# OPE DASHBOARD  (original, unchanged)
# ==============================================================================

if page == "OPE Dashboard":
    st.title("📊 Off-Policy Evaluation (OPE) Dashboard")
    st.caption("IPS · WIS · DM · DR · Bootstrap CI · Policy Gate")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📥 Inputs")
        log_path   = st.text_input("Logged data (.npz)", "data/logged_behavior.npz")
        model_path = st.text_input("Target policy (.pth)", "dqn_policy.pth")

        st.header("⚙️ OPE Settings")
        n_boot   = st.slider("Bootstrap samples", 200, 2000, 800, step=100)
        clip_rho = st.slider("Importance weight clip (ρ)", 5.0, 200.0, 50.0, step=5.0)

        st.header("🚦 Policy Gate")
        baseline = st.number_input("Baseline return", value=0.0, step=10.0)
        margin   = st.number_input("Safety margin",   value=0.0, step=10.0)

        st.header("🧪 CQL Overlay")
        use_cql   = st.checkbox("Compare against CQL", value=False)
        cql_score = st.number_input("CQL score", value=0.0, step=10.0) if use_cql else None

        run_btn = st.button("🚀 Run OPE")

    # Cached execution
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
        seg_ci  = results["segment"]
        rollout = results["rollout"]

        # Deployment Decision Badge
        dr          = overall["DR"]
        deploy_pass = dr["ci_low"] >= baseline + margin

        if deploy_pass:
            st.success("✅ **Deployment decision: PASS** — DR lower bound clears safety gate.")
        else:
            st.error("⛔ **Deployment decision: HOLD** — DR lower bound below safety gate.")

        st.markdown("---")

        # Key Metrics
        st.subheader("📌 Overall OPE Estimates")
        cols = st.columns(4)
        for col, key in zip(cols, ["IPS", "WIS", "DM", "DR"]):
            v = overall[key]
            col.metric(label=key, value=f"{v['estimate']:.2f}",
                       delta=f"[{v['ci_low']:.2f}, {v['ci_high']:.2f}]")

        st.caption(
            f"Episodes={results['meta']['episodes']} · "
            f"ρ_clip={results['meta']['clip_rho']} · "
            f"baseline={baseline} · margin={margin}"
        )
        st.markdown("---")

        # Segment CI
        st.subheader("📊 Segment-wise Confidence Intervals")
        st.pyplot(ci_bar_plot(seg_ci, baseline, cql_score),
                  clear_figure=True, use_container_width=False)
        st.markdown("---")

        # Rollout Guidance
        st.subheader("🧭 Partial Rollout Recommendations")
        rows = []
        for seg in seg_ci:
            rows.append({
                "Segment":          seg,
                "DR Mean":          seg_ci[seg]["estimate"],
                "CI Low":           seg_ci[seg]["ci_low"],
                "CI High":          seg_ci[seg]["ci_high"],
                "Gate":             "PASS" if seg_ci[seg]["ci_low"] >= baseline + margin else "HOLD",
                "Suggested Rollout": rollout[seg]["rollout"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.info("Gate rule: **PASS if CI_low ≥ baseline + margin**")
        st.markdown("---")

        # Sample Distributions
        st.subheader("📈 OPE Sample Distributions")
        st.pyplot(
            dist_plot({
                "IPS": overall["IPS"]["samples"],
                "WIS": overall["WIS"]["samples"],
                "DM":  overall["DM"]["samples"],
                "DR":  overall["DR"]["samples"],
            }),
            clear_figure=True, use_container_width=False,
        )
        st.markdown("---")

        # CQL vs DR
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
