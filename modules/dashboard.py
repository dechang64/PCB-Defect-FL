"""modules/dashboard.py — Landing page with project overview and usage guide.

Reference: monetary-policy-lab/modules/dashboard.py
Goal: 30 seconds to understand what this platform does and how to use it.
"""

import streamlit as st
import json
from pathlib import Path


def render():
    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-header">
        <h1>🔧 Defect-FL · Industrial Defect Detection Platform</h1>
        <p>Federated Learning · YOLO Detection · SAM2 Segmentation · Cross-Factory Collaboration</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode Indicator ──
    if st.session_state.detector.mode == "mock":
        st.info("📊 Currently using **demo mode** with simulated results. Switch to YOLO mode in the sidebar for real inference.")
    else:
        st.success("🔗 **YOLO mode** active — real model inference enabled.")

    # ── Key Metrics ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid #38bdf8;">
            <div style="font-size:0.8rem;color:#94a3b8;">Defect Types</div>
            <div style="font-size:1.8rem;font-weight:700;color:#38bdf8;">6</div>
            <div style="font-size:0.75rem;color:#64748b;">open · short · mousebite · spur · copper · pinhole</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid #8b5cf6;">
            <div style="font-size:0.8rem;color:#94a3b8;">FL Strategies</div>
            <div style="font-size:1.8rem;font-weight:700;color:#8b5cf6;">4</div>
            <div style="font-size:0.75rem;color:#64748b;">FedAvg · FedProx · EWA · TrustFL</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid #22c55e;">
            <div style="font-size:0.8rem;color:#94a3b8;">Student Projects</div>
            <div style="font-size:1.8rem;font-weight:700;color:#22c55e;">16</div>
            <div style="font-size:0.75rem;color:#64748b;">Cohort 2025 (9) + Cohort 2026 (7)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid #f59e0b;">
            <div style="font-size:0.8rem;color:#94a3b8;">Research Tools</div>
            <div style="font-size:1.8rem;font-weight:700;color:#f59e0b;">7</div>
            <div style="font-size:0.75rem;color:#64748b;">Cross-domain · SHAP · SAM2 · Audit · …</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FL Results Preview ──
    st.markdown("### 🌐 Federated Learning Results Preview")
    st.caption("Real FL training on DeepPCB cropped — 3 clients, non-IID 80/20 split, 15 rounds")

    results = _load_fl_results()
    if results:
        col_left, col_right = st.columns(2)

        with col_left:
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
                        "Best Acc": f"{best:.1%}",
                        "Final Acc": f"{final:.1%}",
                        "Best Minority": f"{min_best:.1%}",
                    })

            if "centralized" in results:
                c = results["centralized"]
                table_data.insert(0, {
                    "Strategy": "Centralized (upper bound)",
                    "Best Acc": f"{c['best_acc']:.1%}",
                    "Final Acc": f"{c['final_acc']:.1%}",
                    "Best Minority": "—",
                })

            st.dataframe(table_data, use_container_width=True, hide_index=True)

        with col_right:
            # Convergence chart
            chart_data = {}
            for name in strategies:
                if name in results:
                    d = results[name]
                    chart_data[name] = [r["global_acc"] for r in d]

            if chart_data:
                st.line_chart(chart_data, use_container_width=True)
                st.caption("Global accuracy across communication rounds")
    else:
        st.info("No FL results loaded. Run `fl_training/run_tiny.py` to generate experiment data.")

    # ── Student Highlights ──
    st.markdown("### 🎓 Student Project Highlights")
    st.caption("Top-performing FYP projects under Prof. Dechang Xu")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:1.2rem;border:1px solid #334155;">
            <div style="font-size:0.85rem;font-weight:600;color:#ef4444;">🏆 Best mAP</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-top:0.3rem;">Yuxuan Liu</div>
            <div style="font-size:0.8rem;color:#94a3b8;">YOLOv8n + BiFPN + Attention</div>
            <div style="font-size:1.5rem;font-weight:800;color:#22c55e;margin-top:0.3rem;">98.6% mAP@0.5</div>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:1.2rem;border:1px solid #334155;">
            <div style="font-size:0.85rem;font-weight:600;color:#38bdf8;">⚡ Fastest Edge</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-top:0.3rem;">Wenhao Ma</div>
            <div style="font-size:0.8rem;color:#94a3b8;">Pruned YOLOv8n + TensorRT INT8</div>
            <div style="font-size:1.5rem;font-weight:800;color:#22c55e;margin-top:0.3rem;">238 FPS</div>
        </div>
        """, unsafe_allow_html=True)

    with h3:
        st.markdown("""
        <div style="background:#1e293b;border-radius:10px;padding:1.2rem;border:1px solid #334155;">
            <div style="font-size:0.85rem;font-weight:600;color:#8b5cf6;">💾 Smallest Model</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-top:0.3rem;">Yubo Feng</div>
            <div style="font-size:0.8rem;color:#94a3b8;">YOLOv8n + GhostNet</div>
            <div style="font-size:1.5rem;font-weight:800;color:#22c55e;margin-top:0.3rem;">3.7 MB</div>
        </div>
        """, unsafe_allow_html=True)

    # ── How to Use ──
    st.markdown("---")
    st.markdown("### 🧭 How to Use This Platform")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("""
        **1️⃣ Inspect Defects**
        - Upload a PCB image
        - Detect → Segment → Explain
        - 3-step pipeline in one page
        """)

    with g2:
        st.markdown("""
        **2️⃣ Explore FL Results**
        - Compare FedAvg / FedProx / EWA / TrustFL
        - FedProx μ sensitivity analysis
        - Per-class accuracy breakdown
        """)

    with g3:
        st.markdown("""
        **3️⃣ Research Tools**
        - Cross-domain defect search
        - Transfer feasibility predictor
        - Hallucination defense · SHAP · Audit chain
        """)

    # ── Quick Links ──
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - 📄 [Paper (LaTeX + PDF)](https://github.com/dechang64/PCB-Defect-FL/tree/main/paper)
    - 🏋️ [YOLO Fine-tune Guide](https://github.com/dechang64/PCB-Defect-FL/blob/main/fl_training/YOLO_FINETUNE_GUIDE.md)
    - 🧪 [FL Training Code](https://github.com/dechang64/PCB-Defect-FL/tree/main/fl_training)
    - 📊 [DeepPCB Dataset](https://github.com/dechang64/PCB-Defect-FL/tree/main/results/fl_training/datasets/DeepPCB)
    """)


def _load_fl_results():
    """Load real FL training results from JSON files."""
    RESULTS_DIR = Path(__file__).parent.parent / "results" / "fl_training"
    results = {}
    for f in RESULTS_DIR.glob("cropped_*_rounds.json"):
        name = f.stem.replace("cropped_", "").replace("_rounds", "")
        try:
            results[name] = json.load(open(f))
        except Exception:
            pass
    cent_file = RESULTS_DIR / "cropped_centralized.json"
    if cent_file.exists():
        try:
            results["centralized"] = json.load(open(cent_file))
        except Exception:
            pass
    return results
