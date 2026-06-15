"""feature_search module for Defect-FL."""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time

from utils.constants import DEFECT_CLASSES, SEVERITY_LEVELS



def render():
    st.header("DINOv2 Feature Search")
    st.caption("768-dim self-supervised feature-based PCB similarity retrieval with cross-factory defect pattern matching")

    st.markdown("""
    ### 🔎 How It Works

    ```
    PCB Image → DINOv2 (768-dim) → HNSW Index → Similarity Ranking → Top-K Results
    ```

    **Key Advantages**:
    - No labeled data required (self-supervised learning)
    - Cross-factory defect pattern matching (domain adaptation)
    - Zero-shot defect clustering (discover unknown defect types)
    """)

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.subheader("📤 Query Image")
        query_file = st.file_uploader(
            "Upload query PCB image",
            type=["png", "jpg", "jpeg", "webp"],
            key="search_upload",
        )
        if query_file:
            query_img = Image.open(query_file).convert("RGB")
            st.image(query_img, use_container_width=True, caption="Query Image")

            st.subheader("⚙️ Search Parameters")
            top_k = st.slider("Number of Results", 1, 20, 5)
            sim_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.7, 0.05)

            search_btn = st.button("🔎 Start Search", type="primary", use_container_width=True)

    with col_s2:
        st.subheader("📊 Search Results")
        if query_file and search_btn:
            with st.spinner("🔎 Searching for similar PCBs..."):
                time.sleep(0.5)

                # Mock search results using DINOv2-like features
                np.random.seed(hash(query_file.name) % 2**31)
                query_array = np.array(query_img)

                MOCK_DATABASE = [
                    {"id": "PCB-001", "factory": "Shenzhen SMT Plant", "defect": "short", "similarity": 0.94, "date": "2026-04-20"},
                    {"id": "PCB-002", "factory": "Dongguan PCB Plant", "defect": "open_circuit", "similarity": 0.89, "date": "2026-04-19"},
                    {"id": "PCB-003", "factory": "Suzhou HDI Plant", "defect": "missing_hole", "similarity": 0.85, "date": "2026-04-18"},
                    {"id": "PCB-004", "factory": "Shenzhen SMT Plant", "defect": "spurious_copper", "similarity": 0.82, "date": "2026-04-17"},
                    {"id": "PCB-005", "factory": "Dongguan PCB Plant", "defect": "mouse_bite", "similarity": 0.78, "date": "2026-04-16"},
                    {"id": "PCB-006", "factory": "Suzhou HDI Plant", "defect": "short", "similarity": 0.75, "date": "2026-04-15"},
                    {"id": "PCB-007", "factory": "Shenzhen SMT Plant", "defect": "spur", "similarity": 0.72, "date": "2026-04-14"},
                    {"id": "PCB-008", "factory": "Dongguan PCB Plant", "defect": "open_circuit", "similarity": 0.68, "date": "2026-04-13"},
                ]

                results = [r for r in MOCK_DATABASE if r["similarity"] >= sim_threshold][:top_k]

                if results:
                    for i, r in enumerate(results):
                        sev = SEVERITY_LEVELS.get(r["defect"], {}).get("severity", "minor")
                        sev_icon = "🔴" if sev == "critical" else ("🟡" if sev == "major" else "🟢")
                        st.markdown(f"""
                        **#{i+1} {r['id']}** — {sev_icon} {r['defect']}
                        | Metric | Value |
                        |--------|-------|
                        | Similarity | {r['similarity']:.1%} |
                        | Source Factory | {r['factory']} |
                        | Defect Type | {r['defect']} |
                        | Record Date | {r['date']} |
                        """)
                        st.progress(r["similarity"])
                        if i < len(results) - 1:
                            st.divider()
                else:
                    st.info("No PCB records found above the similarity threshold")
        else:
            st.info("Upload a query image and click search to begin")

    # Feature space visualization
    st.subheader("🌌 Feature Space Visualization")
    st.caption("DINOv2 768-dim features reduced to 2D (t-SNE)")

    np.random.seed(42)
    n_points = 50
    tsne_x = np.random.randn(n_points) * 3
    tsne_y = np.random.randn(n_points) * 3
    # Cluster by defect type
    for i in range(n_points):
        defect_idx = i % 6
        tsne_x[i] += defect_idx * 1.5
        tsne_y[i] += (defect_idx % 3) * 1.5

    chart_data = {
        "x": tsne_x,
        "y": tsne_y,
        "defect": [DEFECT_CLASSES[i % 6] for i in range(n_points)],
    }
    st.scatter_chart(
        chart_data,
        x="x", y="y", color="defect",
        use_container_width=True,
    )
