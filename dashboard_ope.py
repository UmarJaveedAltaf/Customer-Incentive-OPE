# dashboard_ope.py
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

st.title("📊 Off-Policy Evaluation (OPE) Dashboard")
st.caption("IPS · WIS · DM · DR · Bootstrap CI · Policy Gate")

st.markdown("---")


# =============================
# Sidebar
# =============================
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


# =============================
# Cached execution
# =============================
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


# =============================
# Main execution
# =============================
if run_btn:
    with st.spinner("Running off-policy evaluation…"):
        results = cached_run(log_path, model_path, n_boot, baseline, margin, clip_rho)

    overall = results["overall"]
    seg_ci = results["segment"]
    rollout = results["rollout"]

    # =============================
    # 🔔 Deployment Decision Badge
    # =============================
    dr = overall["DR"]
    deploy_pass = dr["ci_low"] >= baseline + margin

    if deploy_pass:
        st.success("✅ **Deployment decision: PASS** — DR lower bound clears safety gate.")
    else:
        st.error("⛔ **Deployment decision: HOLD** — DR lower bound below safety gate.")

    st.markdown("---")

    # =============================
    # 📌 Key Metrics
    # =============================
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

    # =============================
    # 📊 Segment CI Comparison
    # =============================
    st.subheader("📊 Segment-wise Confidence Intervals")
    st.pyplot(
        ci_bar_plot(seg_ci, baseline, cql_score),
        clear_figure=True,
        use_container_width=False
    )

    st.markdown("---")

    # =============================
    # 🧭 Partial Rollout Guidance
    # =============================
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

    # =============================
    # 📈 Sample Distributions
    # =============================
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

    # =============================
    # 🧪 CQL vs DR
    # =============================
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
