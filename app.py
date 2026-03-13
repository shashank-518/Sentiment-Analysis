import streamlit as st
from ml_pipeline import load_artifacts, predict_sentiment

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment AI Pro",
    page_icon="⚡",
    layout="centered",
)

# ── High-End Visual Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark Theme Global Overrides */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }

    /* Main Container Glassmorphism */
    div.block-container {
        padding-top: 2rem;
    }

    /* Input Area Polish */
    .stTextArea textarea {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }

    /* Dynamic Sentiment Cards */
    .sentiment-card {
        padding: 40px;
        border-radius: 24px;
        text-align: center;
        margin: 25px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .positive-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%);
        border-left: 8px solid #22c55e;
    }
    
    .negative-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-left: 8px solid #ef4444;
    }

    .result-label {
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        background: -webkit-linear-gradient(#fff, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .result-conf {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
    }

    /* Modern Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    try:
        return load_artifacts()
    except Exception as e:
        return None, None, None, str(e)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; color: white;'>⚡ Sentiment AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Real-time Recurrent Neural Network analysis</p>", unsafe_allow_html=True)

result_data = get_model()
model, tokenizer, config = result_data[0], result_data[1], result_data[2]

if model is None:
    st.error("Model artifacts not found.")
    st.stop()

# ── Main UI ───────────────────────────────────────────────────────────────────
user_input = st.text_area("Input", placeholder="Enter text to analyze sentiment...", label_visibility="collapsed", height=120)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    analyse_btn = st.button("RUN ANALYSIS")

# ── Example Quick-Actions ─────────────────────────────────────────────────────
st.write("")
cols = st.columns(3)
btns = ["I love this!", "This is terrible.", "Just okay."]
for i, b in enumerate(btns):
    if cols[i].button(b, key=f"b{i}"):
        user_input = b
        analyse_btn = True

# ── Prediction Logic ──────────────────────────────────────────────────────────
if (analyse_btn or user_input) and user_input.strip():
    prediction = predict_sentiment(model, tokenizer, config, user_input.strip())
    sentiment = prediction["sentiment"]
    confidence = prediction["confidence"]
    
    if sentiment == "positive":
        st.markdown(f"""
            <div class="sentiment-card positive-card">
                <p class="result-conf">Analysis Confidence: {confidence*100:.1f}%</p>
                <p class="result-label">POSITIVE</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="sentiment-card negative-card">
                <p class="result-conf">Analysis Confidence: {confidence*100:.1f}%</p>
                <p class="result-label">NEGATIVE</p>
            </div>
        """, unsafe_allow_html=True)

    # ── Probability Metrics ───────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    m1.metric("Positivity", f"{prediction['probabilities']['positive']*100:.1f}%")
    m2.metric("Negativity", f"{prediction['probabilities']['negative']*100:.1f}%")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Engine Specs")
    st.info(f"**Architecture:** SimpleRNN\n\n**Vocabulary:** 20,000\n\n**Status:** Online")
    st.markdown("---")
    st.caption("v2.0.1 | Powered by TensorFlow")