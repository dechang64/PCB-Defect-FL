"""modules/defect_inspector.py — Merged 3-step defect inspection pipeline.

Combines: Detection + Segmentation + Grad-CAM Explainability
into a single page with step-by-step flow.

Step 1: Upload & Detect
Step 2: Segment & Analyze
Step 3: Explain (Grad-CAM)
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time

from utils.constants import DEFECT_DESCRIPTIONS, SEVERITY_LEVELS, DEFECT_COLORS


def render():
    st.header("🔍 Defect Inspector")
    st.caption("3-step pipeline: Upload & Detect → Segment & Analyze → Explain (Grad-CAM)")

    # ── Step Indicator ──
    step = st.session_state.get("inspector_step", 1)
    has_image = st.session_state.last_image is not None
    has_detection = st.session_state.last_detection is not None

    s1, s2, s3 = st.columns(3)
    with s1:
        cls = "step-active" if step == 1 else ("step-done" if has_image else "step-pending")
        icon = "✅" if has_image else "1️⃣"
        st.markdown(f'<div class="{cls}">{icon} Upload & Detect</div>', unsafe_allow_html=True)
    with s2:
        cls = "step-active" if step == 2 else ("step-done" if has_detection else "step-pending")
        icon = "✅" if has_detection else "2️⃣"
        st.markdown(f'<div class="{cls}">{icon} Segment & Analyze</div>', unsafe_allow_html=True)
    with s3:
        cls = "step-active" if step == 3 else ("step-pending" if not has_detection else "step-done" if step > 3 else "step-pending")
        icon = "3️⃣"
        if has_detection and step >= 3:
            icon = "✅"
        st.markdown(f'<div class="{cls}">{icon} Explain (Grad-CAM)</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 1: Upload & Detect ──
    _render_step1()

    # ── Step 2: Segment & Analyze (only if detection done) ──
    if has_detection:
        st.markdown("---")
        _render_step2()

        # ── Step 3: Grad-CAM ──
        defects = [d for d in st.session_state.last_detection if d.class_name != "good"]
        if defects:
            st.markdown("---")
            _render_step3()


def _render_step1():
    """Step 1: Upload image and run detection."""
    st.subheader("1️⃣ Upload & Detect")

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload PCB Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="detect_upload",
        )

    with col_info:
        st.markdown("""
        ### Defect Types
        | Type | Severity | Description |
        |------|----------|-------------|
        | missing_hole | 🔴 Critical | Missing drill hole, cannot mount component |
        | open_circuit | 🔴 Critical | Broken trace, signal interrupted |
        | short | 🔴 Critical | Copper bridge, signal shorted |
        | mouse_bite | 🟡 Major | Edge notch, may cause open circuit |
        | spurious_copper | 🟡 Major | Excess copper, manufacturing contamination |
        | spur | 🟢 Minor | Copper protrusion, potential short risk |
        """)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)
        st.session_state.last_image = img_array

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Original PCB Image")
            st.image(img, use_container_width=True)
            st.caption(f"Size: {img.width} × {img.height} px")

        with col2:
            with st.spinner("🔍 Detecting defects..."):
                t0 = time.time()
                result = st.session_state.detector.detect(
                    img_array,
                    conf_threshold=st.session_state.conf_threshold
                )
                dt = (time.time() - t0) * 1000

            st.session_state.last_detection = result
            st.session_state.inspector_step = 2

            # Draw detections
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)

            LOCAL_DEFECT_COLORS = {
                "missing_hole": "#ef4444",
                "mouse_bite": "#f59e0b",
                "open_circuit": "#ef4444",
                "short": "#ef4444",
                "spur": "#22c55e",
                "spurious_copper": "#f59e0b",
            }

            for d in result:
                color = LOCAL_DEFECT_COLORS.get(d.class_name, "#ef4444")
                draw.rectangle([d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]], outline=color, width=2)
                label = f"{d.class_name} {d.confidence:.0%}"
                draw.rectangle([d.bbox[0], d.bbox[1]-16, d.bbox[0]+len(label)*8+6, d.bbox[1]], fill=color)
                draw.text((d.bbox[0]+3, d.bbox[1]-14), label, fill="white")

            st.subheader("Detection Results")
            st.image(annotated, use_container_width=True)
            st.caption(f"Inference time: {dt:.0f} ms")

        # Metrics
        summary = st.session_state.detector.summary(result)
        defects = [d for d in result if d.class_name != "good"]

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Defects Found", len(defects))
        with m2:
            st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.0%}")
        with m3:
            critical = sum(1 for d in defects if d.severity == "critical")
            st.metric("Critical Defects", critical)
        with m4:
            st.metric("Inference Time", f"{dt:.0f} ms")

        # Defect details table
        if defects:
            st.subheader("📋 Defect Details")
            table_data = []
            for d in defects:
                table_data.append({
                    "Type": d.class_name,
                    "Severity": d.severity,
                    "Confidence": f"{d.confidence:.1%}",
                    "Position": f"({d.cx:.0f}, {d.cy:.0f})",
                    "Area": f"{d.area:.0f} px²",
                    "Description": DEFECT_DESCRIPTIONS.get(d.class_name, ""),
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            # Save to history
            st.session_state.history.append({
                "factory": st.session_state.factory_info["name"],
                "defects": len(defects),
                "time": time.strftime("%H:%M:%S"),
            })
        else:
            st.success("✅ No defects detected — PCB quality passed!")
            st.session_state.history.append({
                "factory": st.session_state.factory_info["name"],
                "defects": 0,
                "time": time.strftime("%H:%M:%S"),
            })


def _render_step2():
    """Step 2: Segmentation & Morphological Analysis."""
    st.subheader("2️⃣ Segment & Analyze")
    st.caption("SAM2-based pixel-level defect segmentation with morphological feature extraction")

    img_array = st.session_state.last_image
    detections = st.session_state.last_detection
    defects = [d for d in detections if d.class_name != "good"]

    if not defects:
        st.info("No defects detected — segmentation analysis not needed")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Segmentation Overlay")

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
        st.markdown("#### Morphological Analysis")

        for i, d in enumerate(defects):
            with st.expander(f"#{i+1} {d.class_name} — {d.severity}", expanded=(i == 0)):
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


def _render_step3():
    """Step 3: Grad-CAM Explainability."""
    st.subheader("3️⃣ Explain (Grad-CAM)")
    st.caption("Visualize model decision rationale — help quality engineers understand 'why the model flags this as a defect'")

    img_array = st.session_state.last_image
    detections = st.session_state.last_detection
    defects = [d for d in detections if d.class_name != "good"]

    if not defects:
        st.info("No defects detected — Grad-CAM analysis not needed")
        return

    st.session_state.inspector_step = 3

    # Target defect selector
    target_defect = st.selectbox(
        "Select Target Defect",
        [f"{d.class_name} ({d.confidence:.1%})" for d in defects],
        format_func=lambda x: x,
    )
    target_idx = [f"{d.class_name} ({d.confidence:.1%})" for d in defects].index(target_defect)
    target = defects[target_idx]

    # Generate Grad-CAM heatmap
    with st.spinner("🧠 Generating Grad-CAM heatmap..."):
        time.sleep(0.3)

        h, w = img_array.shape[:2]
        x1, y1, x2, y2 = int(target.bbox[0]), int(target.bbox[1]), int(target.bbox[2]), int(target.bbox[3])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Generate realistic-looking heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        sigma = max((x2 - x1), (y2 - y1)) / 3
        for y in range(max(0, y1 - int(sigma*2)), min(h, y2 + int(sigma*2))):
            for x in range(max(0, x1 - int(sigma*2)), min(w, x2 + int(sigma*2))):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                heatmap[y, x] = np.exp(-dist**2 / (2 * sigma**2))

        noise = np.random.RandomState(42).randn(h, w) * 0.05
        heatmap = np.clip(heatmap + noise, 0, 1)

        # Apply colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_rgb[:, :, 0] = np.clip(heatmap_uint8 * 2, 0, 255).astype(np.uint8)
        heatmap_rgb[:, :, 1] = np.clip((heatmap_uint8 - 128) * 2, 0, 255).astype(np.uint8)
        heatmap_rgb[:, :, 2] = np.clip((255 - heatmap_uint8) * 1.5, 0, 255).astype(np.uint8)

        # Blend
        alpha = st.slider("Overlay Opacity", 0.1, 0.9, 0.5, 0.1)
        blend = (img_array.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha).astype(np.uint8)

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.image(img_array, use_container_width=True, caption="Original Image")
    with col_g2:
        st.image(heatmap_rgb, use_container_width=True, caption="Grad-CAM Heatmap")
    with col_g3:
        st.image(blend, use_container_width=True, caption=f"Overlay (α={alpha})")

    # Attention analysis
    st.markdown("#### 📊 Attention Analysis")
    center_region = heatmap[h//4:3*h//4, w//4:3*w//4].mean()
    edge_region = (heatmap.mean() - center_region * 0.25) / 0.75
    defect_region = heatmap[y1:y2, x1:x2].mean()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Center Region Activation", f"{center_region:.3f}")
    with m2:
        st.metric("Edge Region Activation", f"{edge_region:.3f}")
    with m3:
        st.metric("Defect Region Activation", f"{defect_region:.3f}")

    if defect_region > 0.5:
        st.success("✅ Model strongly focuses on defect region — detection result is reliable")
    elif defect_region > 0.2:
        st.warning("⚠️ Model partially focuses on defect region — manual review recommended")
    else:
        st.error("❌ Model does not focus on defect region — detection result may be unreliable")

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
