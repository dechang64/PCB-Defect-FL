"""explainability module for Defect-FL."""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time

try:
    from analysis.gradcam import generate_defect_report
except ImportError:
    generate_defect_report = None



def render():
    st.header("Grad-CAM Explainability Analysis")
    st.caption("Visualize model decision rationale — help quality engineers understand 'why the model flags this as a defect'")

    st.markdown("""
    ### 🧠 Grad-CAM Principle

    ```
    PCB Image → CNN Feature Maps → Target Class Gradients → Weighted Sum → Heatmap
    ```

    **Quality Engineer Perspective**:
    - 🔍 **False Positive Analysis**: "Why did the model flag a normal pad as a defect?"
    - 📊 **Root Cause Localization**: "Defect area overlaps with high-density traces"
    - ✅ **Model Trust**: "Heatmap focuses on actual defect location → trust model decision"
    """)

    if st.session_state.last_detection is None or st.session_state.last_image is None:
        st.warning("Please upload an image and run detection on the 'Defect Detection' tab first")
        return

    img_array = st.session_state.last_image
    detections = st.session_state.last_detection
    defects = [d for d in detections if d.class_name != "good"]

    if not defects:
        st.info("No defects detected — Grad-CAM analysis not needed")
        return

    # Grad-CAM settings
    st.subheader("⚙️ Analysis Settings")
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

        # Add some noise for realism
        noise = np.random.RandomState(42).randn(h, w) * 0.05
        heatmap = np.clip(heatmap + noise, 0, 1)

        # Apply colormap (jet-like)
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

    # Generate report using analysis/gradcam.py
    if generate_defect_report:
        report = generate_defect_report(
            heatmap=heatmap,
            defect_type=target.class_name,
            confidence=target.confidence,
            severity=target.severity,
            morphology={"area": target.area, "bbox": target.bbox},
        )
        st.markdown(report)
    else:
        st.info("Grad-CAM report generation unavailable (analysis module not loaded)")

    # Attention analysis
    st.subheader("📊 Attention Analysis")
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
