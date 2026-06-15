"""detection module for Defect-FL."""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time

from utils.constants import DEFECT_DESCRIPTIONS, SEVERITY_LEVELS, DEFECT_COLORS


def render():
    st.header("Industrial Defect Detection (PCB)")
    st.caption("Supports 6 defect types: missing hole, mouse bite, open circuit, short, spur, spurious copper")

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
                result = st.session_state.detector.detect(img_array,
                    conf_threshold=st.session_state.conf_threshold)
                dt = (time.time() - t0) * 1000

            st.session_state.last_detection = result

            # Draw detections
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)

            DEFECT_COLORS = {
                "missing_hole": "#ef4444",
                "mouse_bite": "#f59e0b",
                "open_circuit": "#ef4444",
                "short": "#ef4444",
                "spur": "#22c55e",
                "spurious_copper": "#f59e0b",
            }

            for d in result:
                color = DEFECT_COLORS.get(d.class_name, "#ef4444")
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
            st.markdown(f"""
            <div class="metric-card {'danger' if defects else 'success'}">
                <div style="font-size:2rem;font-weight:700">{len(defects)}</div>
                <div style="color:#666">Defects Found</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;font-weight:700">{summary.get('avg_confidence', 0):.0%}</div>
                <div style="color:#666">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            critical = sum(1 for d in defects if d.severity == "critical")
            st.markdown(f"""
            <div class="metric-card {'danger' if critical else ''}">
                <div style="font-size:2rem;font-weight:700">{critical}</div>
                <div style="color:#666">Critical Defects</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;font-weight:700">{dt:.0f}</div>
                <div style="color:#666">Time (ms)</div>
            </div>
            """, unsafe_allow_html=True)

        # Defect details table
        if defects:
            st.subheader("📋 Defect Details")
            table_data = []
            for d in defects:
                sev_class = f"severity-{d.severity}" if d.severity != "moderate" else "severity-moderate"
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
