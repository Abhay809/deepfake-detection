"""
Streamlit app for DeepFake Detection – professional dashboard UI.
"""

import os
import sys
import streamlit as st
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import load_model, predict_image
from src.preprocessing import get_val_transforms


st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS: professional dashboard ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0c0e12;
        --surface: #161b22;
        --surface-2: #21262d;
        --surface-3: #30363d;
        --border: rgba(255,255,255,0.08);
        --primary: #58a6ff;
        --primary-glow: rgba(88, 166, 255, 0.25);
        --success: #3fb950;
        --success-bg: rgba(63, 185, 80, 0.12);
        --danger: #f85149;
        --danger-bg: rgba(248, 81, 73, 0.12);
        --warning: #d29922;
        --text: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --radius: 12px;
        --radius-lg: 20px;
        --font: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    .stApp {
        background: var(--bg) !important;
        font-family: var(--font) !important;
    }
    .stApp header { background: var(--surface) !important; border-bottom: 1px solid var(--border); }
    .block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px; }

    /* ----- Hero strip ----- */
    .hero-strip {
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 50%, #0d1117 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1.5rem;
    }
    .hero-left { display: flex; align-items: center; gap: 1rem; }
    .hero-icon {
        width: 56px; height: 56px;
        background: linear-gradient(135deg, var(--primary), #1f6feb);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.75rem;
        box-shadow: 0 4px 20px var(--primary-glow);
    }
    .hero-title { font-size: 1.75rem; font-weight: 800; color: var(--text); letter-spacing: -0.03em; margin: 0; }
    .hero-sub { font-size: 0.95rem; color: var(--text-secondary); margin: 0.25rem 0 0 0; }
    .hero-badges {
        display: flex; gap: 0.75rem; flex-wrap: wrap;
    }
    .badge {
        background: var(--surface-3);
        color: var(--text-secondary);
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge.accent { background: rgba(88, 166, 255, 0.15); color: var(--primary); }

    /* ----- Main grid ----- */
    .main-grid {
        display: grid;
        grid-template-columns: 1fr 400px;
        gap: 1.5rem;
        align-items: start;
    }
    @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }

    /* ----- Upload card ----- */
    .upload-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .upload-card h3 { font-size: 1.1rem; font-weight: 700; color: var(--text); margin: 0 0 1rem 0; }
    .upload-zone-custom {
        border: 2px dashed var(--surface-3);
        border-radius: var(--radius);
        padding: 2.5rem;
        text-align: center;
        background: rgba(255,255,255,0.02);
        transition: all 0.2s ease;
    }
    .upload-zone-custom:hover { border-color: var(--primary); background: rgba(88, 166, 255, 0.04); }
    .upload-icon { font-size: 2.5rem; margin-bottom: 0.75rem; opacity: 0.9; }
    .upload-text { color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 0.25rem; }
    .upload-hint { color: var(--text-muted); font-size: 0.8rem; }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploader"] section div[data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
    }

    /* ----- Preview card ----- */
    .preview-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    .preview-card-header {
        padding: 0.75rem 1.25rem;
        background: var(--surface-2);
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        border-bottom: 1px solid var(--border);
    }
    .preview-card img { width: 100%; display: block; }

    /* ----- Result panel (right column) ----- */
    .result-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.75rem;
        position: sticky;
        top: 1.5rem;
    }
    .result-panel.real { border-top: 3px solid var(--success); box-shadow: 0 0 30px var(--success-bg); }
    .result-panel.fake { border-top: 3px solid var(--danger); box-shadow: 0 0 30px var(--danger-bg); }

    .verdict-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.25rem; }
    .verdict-icon {
        width: 52px; height: 52px;
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.75rem;
    }
    .verdict-icon.real { background: var(--success-bg); color: var(--success); }
    .verdict-icon.fake { background: var(--danger-bg); color: var(--danger); }
    .verdict-label { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
    .verdict-label.real { color: var(--success); }
    .verdict-label.fake { color: var(--danger); }

    .gauge-wrap { margin: 1.5rem 0; }
    .gauge-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .gauge-track {
        height: 12px;
        background: var(--surface-2);
        border-radius: 999px;
        overflow: hidden;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .gauge-fill.real { background: linear-gradient(90deg, #238636, var(--success)); }
    .gauge-fill.fake { background: linear-gradient(90deg, #da3633, var(--danger)); }
    .gauge-value { font-size: 1.75rem; font-weight: 800; margin-top: 0.5rem; }
    .gauge-value.real { color: var(--success); }
    .gauge-value.fake { color: var(--danger); }

    .insight-box {
        background: var(--surface-2);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin-top: 1.25rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
        border-left: 3px solid var(--primary);
    }

    /* ----- Sidebar ----- */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }
    .sidebar-section { margin-bottom: 1.5rem; }
    .sidebar-section h4 { color: var(--text); font-size: 0.9rem; margin-bottom: 0.5rem; }
    .sidebar-section ul { margin: 0; padding-left: 1.25rem; color: var(--text-muted); font-size: 0.85rem; line-height: 1.6; }

    /* ----- Empty state ----- */
    .empty-result {
        background: var(--surface-2);
        border: 1px dashed var(--border);
        border-radius: var(--radius-lg);
        padding: 3rem 2rem;
        text-align: center;
        color: var(--text-muted);
    }
    .empty-result .icon { font-size: 2.5rem; margin-bottom: 0.75rem; opacity: 0.6; }

    /* ----- Status pill ----- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: var(--success-bg);
        color: var(--success);
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-pill::before { content: ''; width: 6px; height: 6px; background: var(--success); border-radius: 50%; animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    /* ----- Footer ----- */
    .app-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        text-align: center;
        font-size: 0.8rem;
        color: var(--text-muted);
    }
</style>
""", unsafe_allow_html=True)


# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🛡️ DeepFake Detector")
    st.markdown("")
    st.markdown("""
    <div class="sidebar-section">
        <h4>How it works</h4>
        <ul>
            <li>Upload an image or video</li>
            <li>CNN + transfer learning analyzes the frame</li>
            <li>Frequency & texture artifacts are checked</li>
            <li>You get a Real / Fake verdict with confidence</li>
        </ul>
    </div>
    <div class="sidebar-section">
        <h4>Supported formats</h4>
        <ul>
            <li>Images: JPG, PNG, BMP, WebP</li>
            <li>Video: MP4, AVI, MOV (middle frame used)</li>
        </ul>
    </div>
    <div class="sidebar-section">
        <h4>Model</h4>
        <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">
        EfficientNet backbone with auxiliary artifact features. Train with <code>python run_train.py</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---- Hero strip ----
st.markdown("""
<div class="hero-strip">
    <div class="hero-left">
        <div class="hero-icon">🛡️</div>
        <div>
            <h1 class="hero-title">DeepFake Detection & Prevention</h1>
            <p class="hero-sub">Verify media authenticity with AI-powered artifact analysis</p>
        </div>
    </div>
    <div class="hero-badges">
        <span class="status-pill">Model ready</span>
        <span class="badge accent">CNN + Transfer Learning</span>
        <span class="badge">Frequency Analysis</span>
        <span class="badge">Texture Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---- Model load (with optional download from URL for deployment) ----
def _ensure_checkpoint():
    """If checkpoint is missing and CHECKPOINT_URL is set (e.g. in Streamlit Cloud secrets), download it."""
    checkpoint_path = "checkpoints/best.pt"
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    url = os.environ.get("CHECKPOINT_URL") or (st.secrets.get("CHECKPOINT_URL") if hasattr(st, "secrets") else None)
    if not url:
        return None
    try:
        import urllib.request
        os.makedirs("checkpoints", exist_ok=True)
        urllib.request.urlretrieve(url, checkpoint_path)
        return checkpoint_path if os.path.isfile(checkpoint_path) else None
    except Exception:
        return None


@st.cache_resource
def get_cached_model():
    config_path = "config.yaml"
    checkpoint_path = _ensure_checkpoint()
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/best.pt"
    if not os.path.isfile(checkpoint_path):
        return None, None, None, None, None
    try:
        model, config, device, use_aux = load_model(checkpoint_path, config_path)
        transform = get_val_transforms(config["data"]["image_size"])
        return model, config, device, use_aux, transform
    except Exception:
        return None, None, None, None, None


model, config, device, use_aux, transform = get_cached_model()

if model is None:
    st.error("**Model not found.** Train first: `python run_train.py` with images in `data/real/` and `data/fake/`.")
    st.stop()


# ---- Upload: card + file uploader ----
st.markdown("""
<div class="upload-card">
    <h3>📤 Upload media</h3>
    <div class="upload-zone-custom">
        <div class="upload-icon">📁</div>
        <div class="upload-text">Drag and drop or browse</div>
        <div class="upload-hint">Image or video • JPG, PNG, MP4, AVI, MOV</div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload",
    type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov"],
    label_visibility="collapsed",
)

# ---- Process upload first so we have result before rendering columns ----
img_pil = None
is_real = None
label = None
confidence = None
confidence_pct = None

if uploaded is not None:
    with st.spinner("Analyzing media..."):
        if uploaded.type and "video" in uploaded.type:
            import cv2
            bytes_data = uploaded.read()
            with open("_temp_video.mp4", "wb") as f:
                f.write(bytes_data)
            cap = cv2.VideoCapture("_temp_video.mp4")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
            ret, frame = cap.read()
            cap.release()
            if os.path.exists("_temp_video.mp4"):
                os.remove("_temp_video.mp4")
            if not ret:
                st.error("Could not read video frame.")
                st.stop()
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            temp_path = "_temp_frame.png"
            img_pil.save(temp_path)
            pred, prob_fake = predict_image(temp_path, model, config, device, use_aux, transform)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            img_pil = Image.open(uploaded).convert("RGB")
            temp_path = "_temp_upload.png"
            img_pil.save(temp_path)
            pred, prob_fake = predict_image(temp_path, model, config, device, use_aux, transform)
            if os.path.exists(temp_path):
                os.remove(temp_path)
    is_real = pred == 0
    label = "Real" if is_real else "Fake"
    confidence = (1 - prob_fake) if is_real else prob_fake
    confidence_pct = 100 * confidence

# ---- Two columns: preview (left) | result panel (right) ----
col_preview, col_result = st.columns([1.5, 1])

with col_preview:
    if img_pil is not None:
        st.markdown('<div class="preview-card"><div class="preview-card-header">Preview</div>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    if img_pil is None:
        st.markdown("""
        <div class="result-panel">
            <div class="empty-result">
                <div class="icon">📋</div>
                <strong>Result</strong><br/><br/>
                Upload an image or video to see the authenticity verdict and confidence score.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        panel_class = "real" if is_real else "fake"
        icon = "✓" if is_real else "⚠"
        insight = (
            "No significant manipulation or AI-generation artifacts detected in frequency or texture."
            if is_real else
            "Artifact patterns suggest possible manipulation or AI-generated content. Review source when in doubt."
        )
        st.markdown(f"""
        <div class="result-panel {panel_class}">
            <div class="verdict-row">
                <div class="verdict-icon {panel_class}">{icon}</div>
                <div class="verdict-label {panel_class}">{label}</div>
            </div>
            <div class="gauge-wrap">
                <div class="gauge-label">Confidence</div>
                <div class="gauge-track">
                    <div class="gauge-fill {panel_class}" style="width: {confidence_pct:.1f}%;"></div>
                </div>
                <div class="gauge-value {panel_class}">{confidence:.1%}</div>
            </div>
            <div class="insight-box">{insight}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="app-footer">DeepFake Detection · CNN + artifact analysis · Not a substitute for professional verification</div>', unsafe_allow_html=True)
