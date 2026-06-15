"""fl_training module for Defect-FL — with real experimental results."""

import streamlit as st
import numpy as np
import json
import csv
from pathlib import Path

try:
    from analysis.fl_engine import DefectFLEngine
except ImportError:
    DefectFLEngine = None
from utils.constants import FACTORY_PRESETS, DEFECT_CLASSES

# Load real experiment results
RESULTS_DIR = Path(__file__).parent.parent / "results" / "fl_training"
PHASE2_DIR = Path(__file__).parent.parent / "results" / "phase2_yolo"

def load_real_results():
    """Load real FL training results from JSON files."""
    results = {}
    for f in RESULTS_DIR.glob("cropped_*_rounds.json"):
        name = f.stem.replace("cropped_", "").replace("_rounds", "")
        try:
            results[name] = json.load(open(f))
        except:
            pass
    # Load centralized
    cent_file = RESULTS_DIR / "cropped_centralized.json"
    if cent_file.exists():
        try:
            results["centralized"] = json.load(open(cent_file))
        except:
            pass
    return results


def load_phase2_results():
    """Load Phase 2 YOLOv12n results."""
    p2 = {}
    # Convergence
    conv_file = PHASE2_DIR / "convergence_results.csv"
    if conv_file.exists():
        with open(conv_file) as f:
            p2["convergence"] = list(csv.DictReader(f))
    # μ sensitivity
    mu_file = PHASE2_DIR / "mu_sensitivity.csv"
    if mu_file.exists():
        with open(mu_file) as f:
            p2["mu_sensitivity"] = list(csv.DictReader(f))
    # Round-level data
    for name in ["ewa_rounds", "fedavg_rounds"]:
        rf = PHASE2_DIR / f"{name}.json"
        if rf.exists():
            with open(rf) as f:
                p2[name] = json.load(f)
    # Summary
    sf = PHASE2_DIR / "fl_summary.json"
    if sf.exists():
        with open(sf) as f:
            p2["summary"] = json.load(f)
    return p2


def render():
    st.header("Federated Learning Experiments")
    st.caption("Real FL training on DeepPCB cropped — 3 clients, non-IID 80/20 split, 15 rounds")

    # Load results
    results = load_real_results()
    has_real = len(results) > 0

    # Architecture diagram
    st.markdown("""
    ### 🌐 Federated Learning Architecture

    ```
    Client A (open, short, mousebite dominant)  ──┐
    Client B (spur, copper dominant)             ──┼──→ Aggregation Server ──→ Global Model
    Client C (pinhole dominant)                  ──┘
    ```

    **Dataset**: DeepPCB cropped (8,008 defect crops, 6 classes) | **Model**: CNN48 (4-layer CNN + BN)
    """)

    if not has_real:
        st.warning("No real experiment results found. Run `fl_training/run_tiny.py` first.")
        return

    # ── Tab layout ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Baseline Comparison",
        "🔬 FedProx μ Sensitivity",
        "⚖️ Conformity Penalty",
        "📈 Per-Class Analysis",
        "🚀 Phase 2: YOLOv12n (EWA Active Aggregation)",
    ])

    # ── Tab 1: Baseline Comparison ──
    with tab1:
        st.subheader("FL Baseline Comparison (3 clients, 15 rounds)")

        # Summary table
        strategies = ["FedAvg", "FedProx", "EWA", "TrustFL-Defect"]
        table_data = []
        for name in strategies:
            if name in results:
                d = results[name]
                best = max(r["global_acc"] for r in d)
                final = d[-1]["global_acc"]
                min_best = max(r["global_minority"] for r in d)
                table_data.append({
                    "Strategy": name,
                    "Best Accuracy": f"{best:.1%}",
                    "Final Accuracy": f"{final:.1%}",
                    "Best Minority Acc": f"{min_best:.1%}",
                })

        if "centralized" in results:
            c = results["centralized"]
            table_data.insert(0, {
                "Strategy": "Centralized (upper bound)",
                "Best Accuracy": f"{c['best_acc']:.1%}",
                "Final Accuracy": f"{c['final_acc']:.1%}",
                "Best Minority Acc": "—",
            })

        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # Convergence chart
        st.subheader("Convergence Curves")
        chart_data = {}
        for name in strategies:
            if name in results:
                d = results[name]
                chart_data[name] = [r["global_acc"] for r in d]

        if chart_data:
            st.line_chart(chart_data, use_container_width=True)
            st.caption("Global accuracy across communication rounds")

        # Minority accuracy
        chart_data_min = {}
        for name in strategies:
            if name in results:
                d = results[name]
                chart_data_min[f"{name} (minority)"] = [r["global_minority"] for r in d]

        if chart_data_min:
            st.line_chart(chart_data_min, use_container_width=True)
            st.caption("Minority class accuracy across communication rounds")

    # ── Tab 2: FedProx μ Sensitivity ──
    with tab2:
        st.subheader("FedProx Proximal Term Coefficient (μ) Sensitivity")

        mu_data = {}
        for name in ["FedProx", "FedProx_mu0.1", "FedProx_mu1.0"]:
            if name in results:
                d = results[name]
                label = name.replace("FedProx", "μ=0.01").replace("_mu0.1", "μ=0.1").replace("_mu1.0", "μ=1.0")
                mu_data[label] = [r["global_acc"] for r in d]

        if mu_data:
            st.line_chart(mu_data, use_container_width=True)
            st.caption("FedProx accuracy with different μ values. μ=0.01 achieves the best performance (91.0%).")

        # Summary
        mu_table = []
        for name, label in [("FedProx", "μ=0.01"), ("FedProx_mu0.1", "μ=0.1"), ("FedProx_mu1.0", "μ=1.0")]:
            if name in results:
                d = results[name]
                best = max(r["global_acc"] for r in d)
                mu_table.append({"μ": label, "Best Accuracy": f"{best:.1%}"})

        if mu_table:
            st.dataframe(mu_table, use_container_width=True, hide_index=True)

    # ── Tab 3: Conformity Penalty ──
    with tab3:
        st.subheader("TrustFL-Defect Conformity Penalty Analysis")

        st.markdown("""
        The conformity penalty controls how much minority class suppression reduces a client's aggregation weight.

        - **penalty=1.0**: weight = accuracy × minority_accuracy (full penalty)
        - **penalty=0.2**: weight = accuracy × (1 - 0.2 × (1 - minority_accuracy)) (reduced penalty)
        """)

        if "TrustFL-Defect" in results:
            d = results["TrustFL-Defect"]
            # Show penalty=1.0 data
            p10_acc = [r["global_acc"] for r in d]
            p10_min = [r["global_minority"] for r in d]

            chart = {
                "penalty=1.0 (overall)": p10_acc,
                "penalty=1.0 (minority)": p10_min,
            }
            st.line_chart(chart, use_container_width=True)
            st.caption("TrustFL-Defect with full conformity penalty (penalty=1.0)")

        st.info("""
        **Key Finding**: The full conformity penalty (1.0) outperforms the reduced penalty (0.2) in both
        overall accuracy (87.2% vs 85.3%) and minority class accuracy (82.4% vs 80.2%) after 12 rounds.
        While the reduced penalty provides better initial minority accuracy (56.9% vs 23.1% at Round 1),
        the full penalty's aggressive reweighting produces superior long-term convergence.
        """)

    # ── Tab 4: Per-Class Analysis ──
    with tab4:
        st.subheader("Per-Class Accuracy Comparison")

        # Find best round for each strategy
        for name in strategies:
            if name in results:
                d = results[name]
                best_round = max(d, key=lambda r: r["global_acc"])
                per_class = best_round.get("global_per_class", {})

                if per_class:
                    st.markdown(f"**{name}** (Round {best_round['round']}, acc={best_round['global_acc']:.1%})")
                    class_data = {cls: [acc] for cls, acc in sorted(per_class.items())}
                    st.bar_chart(class_data, use_container_width=True, height=200)

        # Cross-strategy comparison
        st.subheader("Cross-Strategy Per-Class Comparison")
        cross_data = {}
        for name in strategies:
            if name in results:
                d = results[name]
                best_round = max(d, key=lambda r: r["global_acc"])
                per_class = best_round.get("global_per_class", {})
                if per_class:
                    cross_data[name] = [per_class.get(cls, 0) for cls in DEFECT_CLASSES]

        if cross_data:
            chart_df = {name: vals for name, vals in cross_data.items()}
            st.line_chart(chart_df, use_container_width=True)
            st.caption(f"Per-class accuracy across strategies (classes: {', '.join(DEFECT_CLASSES)})")

    # ══════════════════════════════════════════════════════════════
    # Tab 5: Phase 2 — YOLOv12n EWA Active Aggregation
    # ══════════════════════════════════════════════════════════════
    with tab5:
        p2 = load_phase2_results()
        if not p2.get("convergence"):
            st.warning("Phase 2 results not found. Place data in `results/phase2_yolo/`.")
            return

        st.subheader("🚀 Phase 2: YOLOv12n Object Detection + EWA Active Aggregation")
        st.markdown("""
        **Upgrade from CNN → YOLOv12n**: Phase 2 extends FL experiments from image classification (CNN, accuracy) 
        to object detection (YOLOv12n, mAP50-95), and evaluates EWA as an active aggregation strategy.

        | | Phase 1 (CNN) | Phase 2 (YOLOv12n) |
        |---|---|---|
        | **Task** | Image Classification | Object Detection |
        | **Metric** | Accuracy | mAP50-95 |
        | **Data** | DeepPCB cropped (48px) | DeepPCB full (640px) |
        | **Clients** | 3 (2-class dominant) | 3 (2-class specialist) |
        | **Rounds** | 15 | 20 |
        | **Strategies** | FedAvg/FedProx/EWA/TrustFL | FedAvg/FedProx/EWA-v2/EWA-FedProx |
        """)

        conv = p2["convergence"]
        mu_data = p2.get("mu_sensitivity", [])
        ewa_rounds = p2.get("ewa_rounds", [])
        fedavg_rounds = p2.get("fedavg_rounds", [])

        # ── EWA Effective Interval Cards ──
        st.markdown("### 🎯 EWA Effective Interval")
        int_cols = st.columns(3)
        ewa_interval = [
            ("IID Balanced", "iid_balanced", "⚠️ Harmful", "#dc2626"),
            ("Moderate Non-IID", "moderate_balanced", "⚠️ Neutral", "#f59e0b"),
            ("Extreme Non-IID", "extreme_extreme", "✅ Beneficial", "#059669"),
        ]
        for i, (label, scenario, verdict, color) in enumerate(ewa_interval):
            fedavg_match = [r for r in conv if r["scenario"] == scenario and r["strategy"] == "FedAvg"]
            ewa_match = [r for r in conv if r["scenario"] == scenario and r["strategy"] == "EWA-v2"]
            if fedavg_match and ewa_match:
                delta = float(ewa_match[0]["final_mAP"]) - float(fedavg_match[0]["final_mAP"])
                with int_cols[i]:
                    st.metric(
                        label=label,
                        value=f"{delta:+.4f}",
                        delta=f"{delta*100:+.1f}pp vs FedAvg",
                    )
                    st.caption(verdict)

        st.info("""
        **Core Finding**: EWA's benefit is **conditional on data heterogeneity**.
        Harmful under IID (−5.7pp), neutral under moderate Non-IID (−1.3pp),
        beneficial under extreme Non-IID (+1.1pp). This defines the **EWA Effective Interval**.
        """)

        # ── Strategy Comparison ──
        st.markdown("### 📊 Strategy Comparison (Final mAP50-95)")
        strat_names = ["FedAvg", "FedProx", "EWA-v2", "EWA-FedProx"]
        scenario_labels = ["IID Balanced", "Moderate Non-IID", "Extreme Non-IID"]
        scenario_keys = ["iid_balanced", "moderate_balanced", "extreme_extreme"]

        strat_data = {}
        for strat in strat_names:
            vals = []
            for sk in scenario_keys:
                match = [r for r in conv if r["scenario"] == sk and r["strategy"] == strat]
                vals.append(float(match[0]["final_mAP"]) if match else 0)
            strat_data[strat] = vals

        st.bar_chart(strat_data, use_container_width=True)
        st.caption("Final mAP50-95 for each strategy across Non-IID scenarios")

        # ── Final vs Best ──
        st.markdown("### 📉 Final vs Best Performance")
        fb_data = {}
        for strat in strat_names:
            finals, bests = [], []
            for sk in scenario_keys:
                match = [r for r in conv if r["scenario"] == sk and r["strategy"] == strat]
                if match:
                    finals.append(float(match[0]["final_mAP"]))
                    bests.append(float(match[0]["best_mAP"]))
            fb_data[f"{strat} (Final)"] = finals
            fb_data[f"{strat} (Best)"] = bests

        st.line_chart(fb_data, use_container_width=True)
        st.caption("Solid=Final, Dashed=Best. Large gaps indicate training instability.")

        # ── Training Curves ──
        st.markdown("### 📈 Training Curves (Extreme Non-IID)")
        if ewa_rounds or fedavg_rounds:
            curve_data = {}
            if fedavg_rounds:
                curve_data["FedAvg (global mAP)"] = [r["global_mAP"] for r in fedavg_rounds]
            if ewa_rounds:
                curve_data["EWA-v2 (global mAP)"] = [r["global_mAP"] for r in ewa_rounds]
                for ci, cr in enumerate(ewa_rounds[0]["client_results"]):
                    dom = cr.get("dominant_classes", [])
                    curve_data[f"Client {ci} (dom: {dom})"] = [r["client_results"][ci]["mAP"] for r in ewa_rounds]

            st.line_chart(curve_data, use_container_width=True)
            st.caption("Global and per-client mAP50-95 over FL rounds (Extreme Non-IID scenario)")

        # ── Specialist Advantage ──
        if ewa_rounds:
            st.markdown("### 🎯 Specialist Advantage (Dominant vs Non-dominant AP)")
            sa_data = {}
            for ci, cr in enumerate(ewa_rounds[0]["client_results"]):
                dom = cr.get("dominant_classes", [])
                gaps = []
                for r in ewa_rounds:
                    c = r["client_results"][ci]
                    pca = c["per_class_ap"]
                    dom_avg = np.mean([pca[f"class_{dc}"] for dc in dom]) if dom else 0
                    nondom_avg = np.mean([pca[f"class_{j}"] for j in range(6) if j not in dom])
                    gaps.append(dom_avg - nondom_avg)
                sa_data[f"Client {ci} (dom: {dom})"] = gaps

            st.line_chart(sa_data, use_container_width=True)
            st.caption("Positive gap = specialist advantage preserved. EWA protects minority expertise.")

        # ── μ Sensitivity ──
        if mu_data:
            st.markdown("### 🔧 μ Sensitivity (FedProx)")
            mu_scenarios = {"moderate_balanced": "Moderate", "extreme_extreme": "Extreme"}
            mu_chart = {}
            for sk, sl in mu_scenarios.items():
                rows = sorted([r for r in mu_data if r["scenario"] == sk], key=lambda x: float(x["mu"]))
                mu_chart[f"{sl} (Final)"] = [float(r["final_mAP"]) for r in rows]
                mu_chart[f"{sl} (Best)"] = [float(r["best_mAP"]) for r in rows]

            st.line_chart(mu_chart, use_container_width=True)
            st.caption("FedProx mAP50-95 across μ values. Moderate: μ=0.1 best. Extreme: μ=0.001 best for final.")

        # ── Convergence Table ──
        st.markdown("### 📋 Full Results")
        scenario_display = {"iid_balanced": "IID", "moderate_balanced": "Moderate", "extreme_extreme": "Extreme"}
        table_rows = []
        for r in conv:
            table_rows.append({
                "Scenario": scenario_display.get(r["scenario"], r["scenario"]),
                "Strategy": r["strategy"],
                "Final mAP": f"{float(r['final_mAP']):.4f}",
                "Best mAP": f"{float(r['best_mAP']):.4f}",
                "Gap": f"{float(r['best_mAP']) - float(r['final_mAP']):.4f}",
                "Conv Round": r["convergence_round"],
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # ── Export ──
        st.markdown("### 💾 Export")
        col_a, col_b = st.columns(2)
        with col_a:
            csv_lines = ["scenario,strategy,final_mAP,best_mAP,convergence_round,peak_round"]
            for r in conv:
                csv_lines.append(f"{r['scenario']},{r['strategy']},{r['final_mAP']},{r['best_mAP']},{r['convergence_round']},{r['peak_round']}")
            st.download_button("📥 Convergence CSV", "\n".join(csv_lines), "phase2_convergence.csv", "text/csv")
        with col_b:
            json_str = json.dumps(p2, indent=2, ensure_ascii=False)
            st.download_button("📥 All Results JSON", json_str, "phase2_results.json", "application/json")
