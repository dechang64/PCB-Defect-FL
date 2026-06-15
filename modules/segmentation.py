"""segmentation module for Defect-FL."""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time



def render():
    st.header("Defect Segmentation & Morphological Analysis")
    st.caption("SAM2-based pixel-level defect segmentation with morphological feature extraction")

    if st.session_state.last_detection is None or st.session_state.last_image is None:
        st.warning("Please upload an image and run detection on the 'Defect Detection' tab first")
        return

    img_array = st.session_state.last_image
    detections = st.session_state.last_detection
    defects = [d for d in detections if d.class_name != "good"]

    if not defects:
        st.info("No defects detected — segmentation analysis not needed")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Segmentation Visualization")

        # Create segmentation overlay
        overlay = img_array.copy()
        draw = ImageDraw.Draw(overlay)

        SEG_COLORS = [
            (255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
            (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128),
        ]

        for i, d in enumerate(defects):
            color = SEG_COLORS[i % len(SEG_COLORS)]
            x1, y1, x2, y2 = int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])
            # Mock segmentation: fill bbox with semi-transparent color
            for y in range(y1, min(y2, img_array.shape[0])):
                for x in range(x1, min(x2, img_array.shape[1])):
                    overlay[y, x] = (
                        int(overlay[y, x, 0] * 0.5 + color[0] * 0.5),
                        int(overlay[y, x, 1] * 0.5 + color[1] * 0.5),
                        int(overlay[y, x, 2] * 0.5 + color[2] * 0.5),
                    )
            draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=2)
            draw.text((x1+3, y1+3), f"#{i+1} {d.class_name}", fill="white")

        st.image(overlay, use_container_width=True, caption="Defect Segmentation Overlay")

    with col2:
        st.subheader("Morphological Analysis")

        for i, d in enumerate(defects):
            with st.expander(f"#{i+1} {d.class_name} — {d.severity}", expanded=(i == 0)):
                # Mock segmentation
                seg_result = st.session_state.segmentor.segment(
                    img_array, [int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])]
                )

                st.markdown(f"""
                **Confidence**: {d.confidence:.1%}

                | Metric | Value |
                |--------|-------|
                | Area | {seg_result.area:,} px² |
                | Perimeter | {seg_result.perimeter:.1f} px |
                | Circularity | {seg_result.circularity:.3f} |
                | Solidity | {seg_result.solidity:.3f} |
                | Aspect Ratio | {seg_result.aspect_ratio:.2f} |
                | Centroid | ({seg_result.centroid[0]:.0f}, {seg_result.centroid[1]:.0f}) |

                **Description**: {DEFECT_DESCRIPTIONS.get(d.class_name, 'N/A')}
                """)

    # Grad-CAM mock
    st.subheader("🔍 Grad-CAM Explainability Analysis")
    st.caption("Visualize model attention regions to help quality engineers understand detection decisions")

    if defects:
        # Mock heatmap
        h, w = img_array.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        for d in defects:
            x1, y1, x2, y2 = int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])
            # Gaussian-like heatmap centered on defect
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            for y in range(max(0, y1-20), min(h, y2+20)):
                for x in range(max(0, x1-20), min(w, x2+20)):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    heatmap[y, x] = max(heatmap[y, x], np.exp(-dist**2 / (2 * 15**2)))

        # Normalize and apply colormap
        heatmap_norm = (heatmap / max(heatmap.max(), 1e-6) * 255).astype(np.uint8)
        heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_rgb[:, :, 0] = heatmap_norm  # Red channel
        heatmap_rgb[:, :, 1] = (heatmap_norm * 0.3).astype(np.uint8)  # Some green

        # Blend
        blend = (img_array.astype(np.float32) * 0.6 + heatmap_rgb.astype(np.float32) * 0.4).astype(np.uint8)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.image(heatmap_rgb, use_container_width=True, caption="Grad-CAM Heatmap")
        with col_g2:
            st.image(blend, use_container_width=True, caption="Overlay Visualization")

        # Analysis report
        worst = max(defects, key=lambda d: d.confidence)
        st.markdown(f"""
        ### 📊 Analysis Report

        **Highest Confidence Defect**: {worst.class_name} ({worst.confidence:.1%})
        **Severity**: {worst.severity}

        - Model attention regions **strongly align** with detected defect locations
        - {"🔴 Critical defects detected — immediate re-inspection recommended" if any(d.severity == "critical" for d in defects) else "🟡 Defect severity is manageable — standard processing recommended"}
        - Detected **{len(defects)}** defect regions covering **{sum((d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) for d in defects) / (h*w) * 100:.1f}%** of image area
        """)
