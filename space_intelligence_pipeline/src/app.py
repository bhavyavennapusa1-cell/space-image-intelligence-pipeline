"""
Streamlit Mission Control Dashboard for Space Image Intelligence Pipeline.
Real implementation - calls enhancement.py, detection.py, reasoning.py, nasa_api.py directly.
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhancement import ImageEnhancer
from detection import CraterDetector
from reasoning import CraterReasoner

st.set_page_config(
    page_title="Ground-to-Orbit Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# SPACE THEME STYLING
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(1px 1px at 20px 30px, white, transparent),
        radial-gradient(1px 1px at 60px 70px, white, transparent),
        radial-gradient(1.5px 1.5px at 130px 110px, white, transparent),
        radial-gradient(1px 1px at 90px 40px, #a8c5ff, transparent),
        radial-gradient(1.5px 1.5px at 200px 180px, white, transparent),
        radial-gradient(1px 1px at 260px 60px, #a8c5ff, transparent),
        radial-gradient(1px 1px at 310px 150px, white, transparent),
        radial-gradient(2px 2px at 400px 90px, white, transparent),
        radial-gradient(1px 1px at 450px 220px, #a8c5ff, transparent),
        linear-gradient(180deg, #05070f 0%, #0a0e1f 40%, #0d1233 100%);
    background-size: 500px 500px, 500px 500px, 500px 500px, 500px 500px,
                     500px 500px, 500px 500px, 500px 500px, 500px 500px,
                     500px 500px, cover;
    background-attachment: fixed;
    color: #dce3f5;
}

h1, h2, h3, h4 { color: #f0f3ff !important; font-weight: 600 !important; }

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7db8ff, #b39dff, #ff9ecd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-subtitle {
    color: #8b93b5;
    font-size: 1rem;
    margin-top: 4px;
    letter-spacing: 0.3px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 14px;
    padding: 20px;
    backdrop-filter: blur(6px);
}

.metric-glass {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 18px 10px;
    text-align: center;
    transition: 0.2s;
}
.metric-glass:hover { border-color: rgba(125, 184, 255, 0.5); }
.metric-number { font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-label { color: #8b93b5; font-size: 0.8rem; letter-spacing: 0.5px; text-transform: uppercase; margin-top: 4px; }

.det-high {
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-left: 3px solid #34d399;
    padding: 10px 14px;
    border-radius: 8px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}
.det-low {
    background: rgba(250, 204, 21, 0.08);
    border: 1px solid rgba(250, 204, 21, 0.3);
    border-left: 3px solid #facc15;
    padding: 10px 14px;
    border-radius: 8px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}

.analysis-box {
    background: linear-gradient(135deg, rgba(125,184,255,0.08), rgba(179,157,255,0.08));
    border: 1px solid rgba(125,184,255,0.25);
    border-radius: 14px;
    padding: 22px;
    line-height: 1.6;
    font-size: 0.98rem;
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border-radius: 8px 8px 0 0;
    color: #8b93b5;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    background: rgba(125,184,255,0.15) !important;
    color: #f0f3ff !important;
}

section[data-testid="stSidebar"] {
    background: rgba(5, 7, 15, 0.6);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.stFileUploader {
    border: 1.5px dashed rgba(125,184,255,0.4) !important;
    border-radius: 14px !important;
}

div[data-testid="stStatusWidget"] { color: #7db8ff; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.markdown('<p class="hero-title">🛰️ Ground-to-Orbit Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered crater detection & grounded surface analysis for lunar and Martian imagery</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Mission Controls")
    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.25, 0.05)
    st.markdown("---")
    st.markdown("### 📡 Pipeline Stages")
    st.markdown("""
    1. **Enhance** — Real-ESRGAN 4x super-resolution
    2. **Detect** — Fine-tuned YOLOv8 crater detection
    3. **Reason** — Gemini-grounded plain-English analysis
    4. **Interact** — Ask follow-up questions
    """)
    st.markdown("---")
    st.caption("Built for SIH internal hackathon · Person D module")

# ---------------------------------------------------------------
# MODEL LOADING (cached so it only runs once)
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    enhancer = ImageEnhancer(model_path="weights/RealESRGAN_x4plus.pth")
    detector = CraterDetector(model_path="weights/crater_best_v2.pt")
    reasoner = CraterReasoner(api_key=os.environ.get("GEMINI_API_KEY"))
    return enhancer, detector, reasoner

try:
    enhancer, detector, reasoner = load_models()
    models_ready = True
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    models_ready = False

# ---------------------------------------------------------------
# MAIN UPLOAD FLOW
# ---------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a lunar or Martian surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and models_ready:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    tab1, tab2, tab3 = st.tabs(["🔍  Enhance & Detect", "🧠  AI Analysis", "💬  Ask About This Image"])

    with st.spinner("🛰️ Enhancing image resolution..."):
        enhanced_path = enhancer.enhance("uploaded_image.jpg", "enhanced_output.jpg")

    with st.spinner("🎯 Scanning for surface features..."):
        detection_result = detector.detect("enhanced_output.jpg", conf_threshold=conf_threshold)

    # --- TAB 1: Enhance & Detect ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image("uploaded_image.jpg", use_container_width=True)
        with col2:
            st.markdown("**Enhanced (Real-ESRGAN 4x)**")
            st.image(enhanced_path, use_container_width=True)

        st.markdown("#### Detection Overlay")
        st.image(detection_result["annotated_image_path"], use_container_width=True)

        high_conf = sum(1 for d in detection_result["detections"] if d["confidence"] >= 0.5)
        low_conf = detection_result["count"] - high_conf

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-glass"><div class="metric-number">{detection_result["count"]}</div><div class="metric-label">Total Detected</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-glass"><div class="metric-number" style="color:#34d399">{high_conf}</div><div class="metric-label">High Confidence</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-glass"><div class="metric-number" style="color:#facc15">{low_conf}</div><div class="metric-label">Needs Review</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Individual Detections")
        for d in detection_result["detections"]:
            css_class = "det-high" if d["confidence"] >= 0.5 else "det-low"
            icon = "🟢" if d["confidence"] >= 0.5 else "🟡"
            st.markdown(
                f'<div class="{css_class}">{icon} <b>{d["label"]}</b> — confidence {d["confidence"]*100:.1f}% — box {d["box"]}</div>',
                unsafe_allow_html=True
            )

    # --- TAB 2: AI Analysis ---
    with tab2:
        with st.spinner("🧠 Generating grounded analysis..."):
            analysis = reasoner.reason(detection_result)
        st.markdown("#### Mission Analysis Summary")
        st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)

    # --- TAB 3: Interactive Q&A ---
    with tab3:
        st.markdown("#### Ask a question about this image")
        user_question = st.text_input("e.g. 'Is this a safe area to land a rover?'", key="qa_input")
        if user_question:
            with st.spinner("🛰️ Analyzing..."):
                followup_prompt = f"""Based on this crater detection data: {detection_result['detections']}
Answer this question in 2-3 sentences, staying strictly grounded in the actual data above,
and explicitly note if any answer depends on a low-confidence (below 0.5) detection: {user_question}"""
                response = reasoner.client.models.generate_content(
                    model=reasoner.model, contents=followup_prompt
                )
            st.markdown(f'<div class="analysis-box">{response.text}</div>', unsafe_allow_html=True)

elif not models_ready:
    st.warning("Models could not be loaded. Check that weights files and API keys are correctly configured.")
else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 60px 20px;">
        <h3>👆 Upload an image above to begin analysis</h3>
        <p style="color:#8b93b5;">Supports lunar and Martian surface photography (JPG, PNG)</p>
    </div>
    """, unsafe_allow_html=True)
