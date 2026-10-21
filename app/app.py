"""Streamlit web application for the Plant Disease Detector.

Run:
    pip install -e .
    streamlit run app/app.py
"""

import io
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

matplotlib.use("Agg")

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.dataset import CLASSES, get_transforms, prepare_input_image
from src.diseases import (
    DISCLAIMER,
    SEVERITY_COLOR,
    SEVERITY_LABEL,
    get_disease_info,
)
from src.gradcam import GradCAM, explain
from src.model import EfficientNetB4Classifier
from src.utils import get_device, load_checkpoint

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants & Sample Demos
# ------------------------------------------------------------------

CHECKPOINT_PATH = _project_root / "models" / "best_model.pth"
SAMPLES_DIR = _project_root / "samples"
NUM_CLASSES = 38
IMG_SIZE = 224

SAMPLE_IMAGES = {
    "🍎 Apple — Black Rot": SAMPLES_DIR / "apple_black_rot.jpg",
    "🍇 Grape — Black Measles": SAMPLES_DIR / "grape_black_measles.jpg",
    "🍅 Tomato — Late Blight": SAMPLES_DIR / "tomato_late_blight.jpg",
    "🌽 Corn — Leaf Blight": SAMPLES_DIR / "corn_leaf_blight.jpg",
}

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detector · AI Pathology",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS - Sleek Dark Cyber-Bio Theme with Smooth Animations
# ------------------------------------------------------------------

st.markdown("""
<style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ambient glowing background aura */
    .stApp {
        background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.08) 0%, rgba(11, 15, 25, 0.95) 45%, #0b0f19 100%) !important;
        background-attachment: fixed;
    }

    /* Hero Title with Shimmering Gradient Animation */
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
        padding: 1.5rem 0 0.5rem 0;
    }

    .hero-title {
        font-size: 2.9rem;
        font-weight: 800;
        background: linear-gradient(120deg, #10b981 0%, #34d399 25%, #06b6d4 50%, #10b981 75%, #a7f3d0 100%);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 7s ease infinite;
        letter-spacing: -0.025em;
        margin: 0 auto 0.4rem auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 0.5rem;
    }

    @keyframes floatAnim {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-8px) rotate(4deg); }
    }

    .floating-leaf {
        display: inline-block;
        animation: floatAnim 3.8s ease-in-out infinite;
        filter: drop-shadow(0 0 16px rgba(16, 185, 129, 0.6));
    }

    .hero-sub {
        text-align: center !important;
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 680px;
        margin: 0 auto 1.2rem auto !important;
        line-height: 1.6;
        display: block;
    }

    /* Hero Badges */
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }

    .hero-badge {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    .hero-badge:hover {
        background: rgba(16, 185, 129, 0.25);
        border-color: #10b981;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }

    /* AI Bio-Scanner Laser Preview Container */
    .scanner-wrapper {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5), 0 0 20px rgba(16, 185, 129, 0.15);
        background: #0d131f;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .scanner-wrapper:hover {
        border-color: rgba(16, 185, 129, 0.7);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(16, 185, 129, 0.3);
    }

    @keyframes scanSweep {
        0% { top: 0%; opacity: 0.2; }
        15% { opacity: 1; }
        85% { opacity: 1; }
        100% { top: 98%; opacity: 0.2; }
    }

    .scan-laser-line {
        position: absolute;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #10b981 30%, #34d399 50%, #10b981 70%, transparent 100%);
        box-shadow: 0 0 12px #10b981, 0 0 24px rgba(52, 211, 153, 0.9);
        animation: scanSweep 3s ease-in-out infinite alternate;
        z-index: 10;
        pointer-events: none;
    }

    .scanner-status {
        position: absolute;
        top: 12px;
        right: 12px;
        background: rgba(11, 15, 25, 0.88);
        border: 1px solid rgba(16, 185, 129, 0.6);
        color: #34d399;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 8px;
        z-index: 15;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 6px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    @keyframes blinkDot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.3; transform: scale(0.7); }
    }

    .scanner-dot {
        width: 7px;
        height: 7px;
        background: #10b981;
        border-radius: 50%;
        animation: blinkDot 1.4s ease-in-out infinite;
        box-shadow: 0 0 8px #10b981;
    }

    /* Prediction Result Cards & Severity Glows */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseHealthy {
        0%, 100% {
            box-shadow: 0 0 16px rgba(16, 185, 129, 0.25), inset 0 0 20px rgba(16, 185, 129, 0.08);
            border-color: rgba(16, 185, 129, 0.6);
        }
        50% {
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.5), inset 0 0 25px rgba(16, 185, 129, 0.2);
            border-color: #10b981;
        }
    }

    @keyframes pulseModerate {
        0%, 100% {
            box-shadow: 0 0 16px rgba(245, 158, 11, 0.25), inset 0 0 20px rgba(245, 158, 11, 0.08);
            border-color: rgba(245, 158, 11, 0.6);
        }
        50% {
            box-shadow: 0 0 30px rgba(245, 158, 11, 0.5), inset 0 0 25px rgba(245, 158, 11, 0.2);
            border-color: #f59e0b;
        }
    }

    @keyframes pulseSevere {
        0%, 100% {
            box-shadow: 0 0 18px rgba(239, 68, 68, 0.25), inset 0 0 20px rgba(239, 68, 68, 0.08);
            border-color: rgba(239, 68, 68, 0.6);
        }
        50% {
            box-shadow: 0 0 32px rgba(239, 68, 68, 0.6), inset 0 0 25px rgba(239, 68, 68, 0.25);
            border-color: #ef4444;
        }
    }

    .pred-card {
        padding: 1.3rem 1.6rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 1.1rem;
        backdrop-filter: blur(12px);
        animation: fadeInUp 0.4s ease-out;
        transition: all 0.3s ease;
    }

    .pred-healthy  {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 2px solid rgba(16, 185, 129, 0.5);
        animation: pulseHealthy 3s infinite ease-in-out;
    }

    .pred-moderate {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 2px solid rgba(245, 158, 11, 0.5);
        animation: pulseModerate 2.6s infinite ease-in-out;
    }

    .pred-severe   {
        background: rgba(239, 68, 68, 0.14);
        color: #f87171;
        border: 2px solid rgba(239, 68, 68, 0.5);
        animation: pulseSevere 2.2s infinite ease-in-out;
    }

    /* Metric & Risk Cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        flex: 1;
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
    }

    .metric-title {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Animated Top-5 Custom Prediction Progress Bars */
    .progress-item {
        margin-bottom: 0.75rem;
        animation: fadeInUp 0.45s ease-out;
    }

    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.35rem;
    }

    .progress-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        color: #34d399;
        font-weight: 700;
    }

    .progress-track {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.07);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #059669 0%, #10b981 50%, #34d399 100%);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        transition: width 0.9s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Disclaimer Card */
    .disclaimer-banner {
        background: rgba(245, 158, 11, 0.07);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 0.85rem 1.15rem;
        font-size: 0.86rem;
        color: #fef08a;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 0.3s ease-out;
        line-height: 1.5;
    }

    /* Treatment Box */
    .treatment-box {
        background: rgba(17, 24, 39, 0.8);
        border-left: 4px solid #10b981;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.1rem 1.35rem;
        font-size: 0.94rem;
        line-height: 1.65;
        color: #e2e8f0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
        animation: fadeInUp 0.5s ease-out;
    }

    /* Tag Chips with interactive hover */
    .chip {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.09);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #34d399;
        margin: 3px 4px;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .chip:hover {
        background: rgba(16, 185, 129, 0.25);
        border-color: #10b981;
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        color: #a7f3d0;
    }

    /* Glass card container */
    .glass-panel {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .glass-panel:hover {
        border-color: rgba(16, 185, 129, 0.3);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), 0 0 15px rgba(16, 185, 129, 0.1);
        transform: translateY(-2px);
    }

    /* Sample Selector Pill Header */
    .sample-intro {
        font-size: 0.9rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    /* Streamlit input & file uploader styling */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(16, 185, 129, 0.35) !important;
        border-radius: 14px !important;
        background: rgba(17, 24, 39, 0.55) !important;
        padding: 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #10b981 !important;
        background: rgba(16, 185, 129, 0.05) !important;
        box-shadow: 0 0 18px rgba(16, 185, 129, 0.2) !important;
    }

    /* Button hover improvements */
    .stButton > button {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.2) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        color: #34d399 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #0b0f19 !important;
        border-color: #34d399 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35) !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 1.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Model loader (cached)
# ------------------------------------------------------------------


@st.cache_resource(show_spinner="Initializing EfficientNet-B4 neural engine ...")
def load_model():
    """Load the classifier from the local checkpoint.

    Streamlit does not cache exceptions, so a failure here is retried on the
    next run rather than remembered.
    """
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Please place best_model.pth in the models/ directory."
        )
    device = get_device()
    model = EfficientNetB4Classifier(num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, str(CHECKPOINT_PATH), device)
    return model.to(device).eval(), device


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def analyse(model, device, image_pil: Image.Image):
    """Run the model once and return (probs, cam, denormalised image, pred_idx).

    Grad-CAM already performs a forward pass to build the graph it backpropagates
    through, so its softmax is reused instead of running inference a second time.
    """
    tensor = get_transforms("val", img_size=IMG_SIZE)(image_pil)
    cam, img_np, pred_idx, probs = explain(model, tensor, device)
    return probs, cam, img_np, pred_idx


def build_figure(image_pil, cam, img_np, overlay):
    """Build a high-contrast dark-themed diagnostic panel."""
    BG = "#0b0f19"
    fig = plt.figure(figsize=(16, 5), facecolor=BG)

    panels = [
        (np.array(image_pil), "1. Original Leaf", None),
        (cam, "2. Grad-CAM Activation Heatmap", "hot"),
        (overlay, "3. Diagnostic AI Overlay", None),
    ]
    for i, (img, title, cmap) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, i)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        ax.set_facecolor(BG)

    fig.tight_layout(pad=1.2)
    return fig


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

def sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom: 0.5rem;">
                <span style="font-size: 1.6rem;">🌿</span>
                <span style="font-size: 1.25rem; font-weight: 800; color: #f1f5f9;">Diagnostics Hub</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "**Model:** EfficientNet-B4 (transfer learning)\n\n"
            "**Dataset:** PlantVillage (87,000+ images)\n\n"
            "**Classes:** 38 (14 plant species)\n\n"
            "**Diseases:** 26 unique pathogens\n\n"
            "**Explainability:** Grad-CAM heatmaps"
        )

        st.markdown("### ⚡ Tech Stack")
        tags = ["PyTorch", "EfficientNet-B4", "Grad-CAM", "Streamlit", "OpenCV", "scikit-learn"]
        st.markdown(
            " ".join(f'<span class="chip">{t}</span>' for t in tags),
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 Supported Species")
        plants = sorted({c.split("___")[0].replace("_", " ") for c in CLASSES})
        st.markdown(
            " ".join(f'<span class="chip">{p}</span>' for p in plants),
            unsafe_allow_html=True,
        )

        st.markdown("### 🔍 Workflow")
        st.markdown(
            "1. **Upload or Select** a clear photo of a single leaf\n"
            "2. **AI Bio-Scanner** runs neural classification\n"
            "3. **Grad-CAM** locates pathogenic regions\n"
            "4. **Agronomic Advice** details care & treatment"
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="color: #64748b; font-size: 0.78rem; text-align: center;">
                Built by <strong style="color: #94a3b8;">ARAVINDHAN R</strong><br/>
                EfficientNet-B4 + Grad-CAM Explainability
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------

def main():
    # Hero Title with Animation & Badges
    st.markdown(
        """
        <div class="hero-container" style="text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <h1 class="hero-title" style="text-align: center; justify-content: center;">
                <span class="floating-leaf">🌿</span> Plant Disease Detector
            </h1>
            <p class="hero-sub" style="text-align: center; margin-left: auto; margin-right: auto;">
                Real-time deep learning leaf pathology scanner powered by 
                <strong style="color: #34d399;">EfficientNet-B4</strong> and visualised with <strong style="color: #34d399;">Grad-CAM</strong> attention maps.
            </p>
            <div class="hero-badges" style="justify-content: center;">
                <span class="hero-badge">⚡ EfficientNet-B4 Architecture</span>
                <span class="hero-badge">🔬 Grad-CAM Explainability</span>
                <span class="hero-badge">🎯 38 Disease Classes</span>
                <span class="hero-badge">🛡️ Severity Assessment</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar()

    try:
        model, device = load_model()
    except Exception as exc:
        logger.exception("Model load failed")
        st.error(
            "**Could not load the model.**\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Please ensure `best_model.pth` is in the `models/` directory. "
            "Retry below."
        )
        if st.button("🔄 Retry loading model"):
            load_model.clear()
            st.rerun()
        return

    # ---- 1-Click Interactive Demo Samples ----
    st.markdown(
        '<p class="sample-intro">⚡ Quick Test: Click a sample leaf below or upload your own photo</p>',
        unsafe_allow_html=True,
    )

    demo_cols = st.columns(len(SAMPLE_IMAGES) + 1)
    for col, (label, sample_path) in zip(demo_cols[:-1], SAMPLE_IMAGES.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["active_sample"] = str(sample_path)
                st.session_state["sample_label"] = label

    with demo_cols[-1]:
        if st.session_state.get("active_sample") and st.button("🧹 Clear", use_container_width=True):
            st.session_state.pop("active_sample", None)
            st.session_state.pop("sample_label", None)
            st.rerun()

    uploaded = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"],
        help="Clear, close-up photo of a single leaf — JPEG or PNG",
    )

    raw_image = None
    source_description = ""

    if uploaded is not None:
        try:
            raw_image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            source_description = f"Uploaded photo: `{uploaded.name}`"
        except Exception as exc:
            st.error(f"Could not read that image: `{type(exc).__name__}: {exc}`")
            return
    elif st.session_state.get("active_sample"):
        sample_file = Path(st.session_state["active_sample"])
        if sample_file.exists():
            raw_image = Image.open(sample_file).convert("RGB")
            source_description = f"Demo sample: {st.session_state.get('sample_label', sample_file.name)}"

    if raw_image is None:
        st.markdown(
            """
            <div class="glass-panel" style="text-align: center; padding: 2rem 1rem;">
                <span style="font-size: 2.2rem;">🍃</span>
                <h3 style="color: #f1f5f9; margin-top: 0.5rem; margin-bottom: 0.3rem;">Ready for Diagnostics</h3>
                <p style="color: #94a3b8; font-size: 0.95rem; max-width: 480px; margin: 0 auto;">
                    Upload a leaf photograph above or click any of the <strong>4 Quick Test samples</strong> to activate the AI Bio-Scanner.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _show_class_grid()
        return

    # Scale-and-centre-crop rather than squashing: preserves aspect ratio
    image_pil = prepare_input_image(raw_image, IMG_SIZE)

    with st.spinner("🔬 AI Bio-Scanner analyzing leaf pathology..."):
        probs, cam, img_np, pred_idx = analyse(model, device, image_pil)
        overlay = GradCAM.overlay(cam, img_np, alpha=0.45)

    top5_idx = np.argsort(probs)[::-1][:5]
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    info = get_disease_info(pred_class)
    severity = info["severity"]

    # ---- Agronomic Disclaimer Banner ----
    st.markdown(
        f"""
        <div class="disclaimer-banner">
            <span style="font-size: 1.15rem; margin-top: -2px;">⚠️</span>
            <div>
                <strong>Advisory:</strong> {DISCLAIMER}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Top Result Strip ----
    col_img, col_pred = st.columns([1, 1.4], gap="large")

    with col_img:
        st.markdown(
            f"""
            <div class="scanner-wrapper">
                <div class="scan-laser-line"></div>
                <div class="scanner-status">
                    <span class="scanner-dot"></span> SCAN COMPLETE
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(image_pil, caption=f"Analyzed Leaf · {source_description}", use_container_width=True)

    with col_pred:
        css_cls = {
            "none": "pred-healthy",
            "moderate": "pred-moderate",
            "severe": "pred-severe",
        }.get(severity, "pred-moderate")

        st.markdown(
            f'<div class="pred-card {css_cls}">{info["icon"]}  '
            f'{info["plant"]} — {info["label"]}</div>',
            unsafe_allow_html=True,
        )

        sev_color = SEVERITY_COLOR.get(severity, "#8e44ad")
        sev_label = SEVERITY_LABEL.get(severity, "Unknown")

        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-title">Risk Assessment</div>
                    <div class="metric-value" style="color: {sev_color};">{sev_label}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Diagnostic Confidence</div>
                    <div class="metric-value" style="color: #10b981;">{confidence:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if info["pathogen"]:
            st.markdown(
                f'<div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.9rem;">'
                f'Identified Pathogen: <em style="color: #e2e8f0; font-weight: 600;">{info["pathogen"]}</em></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="font-size: 0.92rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.6rem;">'
            '📊 Top-5 Predictions</div>',
            unsafe_allow_html=True,
        )

        for idx in top5_idx:
            other = get_disease_info(CLASSES[idx])
            other_label = f"{other['plant']} — {other['label']}"
            prob_pct = probs[idx] * 100
            st.markdown(
                f"""
                <div class="progress-item">
                    <div class="progress-header">
                        <span>{other["icon"]} {other_label}</span>
                        <span class="progress-pct">{prob_pct:.1f}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {prob_pct:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Treatment & Explainability Section ----
    st.divider()
    col_treat, col_cam = st.columns([1, 1.4], gap="large")

    with col_treat:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom: 0.6rem;">
                <span style="font-size: 1.4rem;">{info['icon']}</span>
                <span style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9;">Care & Treatment</span>
            </div>
            <div class="treatment-box">{info["treatment"]}</div>
            """,
            unsafe_allow_html=True,
        )

    with col_cam:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom: 0.3rem;">
                <span style="font-size: 1.4rem;">🔬</span>
                <span style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9;">Grad-CAM Explainability</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 0.8rem;">
                The attention heatmap reveals the specific leaf areas driving the neural decision.
                <span style="color: #ef4444; font-weight: 600;">Warm colours (red/yellow)</span> indicate peak feature activation.
            </p>
            """,
            unsafe_allow_html=True,
        )
        fig = build_figure(image_pil, cam, img_np, overlay)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _show_class_grid():
    """Show a styled interactive summary grid of all 38 supported classes."""
    st.divider()
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom: 1rem;">
            <span style="font-size: 1.4rem;">📋</span>
            <span style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9;">Supported Plant Taxonomy & Classes (38)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    plants: dict[str, list[str]] = {}
    for cls in CLASSES:
        info = get_disease_info(cls)
        plants.setdefault(info["plant"], []).append(info["label"])

    cols = st.columns(3)
    for i, (plant, diseases) in enumerate(sorted(plants.items())):
        with cols[i % 3]:
            healthy = [d for d in diseases if "healthy" in d.lower()]
            sick = [d for d in diseases if "healthy" not in d.lower()]

            badges = (
                [f'<span class="chip" style="color: #34d399;">✅ {d}</span>' for d in healthy]
                + [f'<span class="chip" style="color: #fbbf24; border-color: rgba(245, 158, 11, 0.35);">⚠️ {d}</span>' for d in sick]
            )

            st.markdown(
                f"""
                <div class="glass-panel" style="margin-bottom: 0.8rem;">
                    <div style="font-weight: 700; color: #34d399; margin-bottom: 0.4rem; font-size: 0.95rem;">
                        🌱 {plant}
                    </div>
                    <div>{" ".join(badges)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
