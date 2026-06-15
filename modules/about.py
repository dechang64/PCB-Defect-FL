"""about module for Defect-FL."""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time



def render():
    st.header("About Defect-FL")

    st.markdown("""
    ### 🔧 Project Overview

    **Defect-FL** is a federated learning-based intelligent industrial defect detection platform for PCBs, enabling multi-factory collaborative training while keeping PCB image data on-premise.

    ### 🔬 Core Technologies

    | Technology | Purpose |
    |------------|---------|
    | YOLOv11 | Real-time 6-class industrial defect detection (PCB) |
    | SAM2 | Pixel-level defect segmentation |
    | DINOv2 | 768-dim PCB feature extraction |
    | Grad-CAM | Model decision explainability |
    | FedAvg | Multi-factory federated aggregation |
    | HNSW | Defect pattern similarity search |

    ### 🏭 Supported Factories

    | Factory | Production Lines | Daily Capacity |
    |---------|-----------------|----------------|
    | Shenzhen SMT Plant | 8 | 50,000 boards |
    | Dongguan PCB Plant | 5 | 30,000 boards |
    | Suzhou HDI Plant | 3 | 15,000 boards |

    ### 📊 Detection Capabilities

    - Defect Types: 6 classes (missing hole / mouse bite / open circuit / short / spur / spurious copper)
    - Detection Accuracy: >95% (YOLOv11)
    - Inference Speed: <50ms/image
    - Segmentation: Pixel-level (SAM2)

    ### 📄 License

    Apache-2.0 | [GitHub](https://github.com/dechang64)
    """)
