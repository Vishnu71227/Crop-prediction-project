import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

CROP_NAMES = np.array([
    'apple', 'banana', 'blackgram', 'chickpea',
    'coconut', 'coffee', 'cotton', 'grapes',
    'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon',
    'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
])

# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('crop_recommendation_model.pkl')
    le    = joblib.load('crop_label_encoder.pkl')
    return model, le

model, le = load_model()

# ── Header ─────────────────────────────────────────────────
st.title("🌾 Crop Recommendation System")
st.markdown("#### AI-powered crop suggestion based on soil & climate data")
st.markdown("---")

# ── Info Cards ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("🌱 Total Crops", "22")
col2.metric("📊 Model Accuracy", "99.3%")
col3.metric("🧪 Dataset Size", "2200 samples")

st.markdown("---")

# ── Input Section ──────────────────────────────────────────
st.subheader("📋 Enter Soil & Climate Details")
st.markdown("*Sliders se values set karo ya directly number type karo*")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**🧪 Soil Nutrients**")
    N = st.slider("Nitrogen (N) — mg/kg", min_value=0, max_value=140, value=90, help="Nitrogen content in soil")
    P = st.slider("Phosphorus (P) — mg/kg", min_value=5, max_value=145, value=42, help="Phosphorus content in soil")
    K = st.slider("Potassium (K) — mg/kg", min_value=5, max_value=205, value=43, help="Potassium content in soil")
    ph = st.slider("pH Level", min_value=3.5, max_value=10.0, value=6.5, step=0.1, help="pH value of soil")

with col_right:
    st.markdown("**🌤️ Climate Conditions**")
    temperature = st.slider("Temperature (°C)", min_value=8.0, max_value=44.0, value=21.0, step=0.1)
    humidity    = st.slider("Humidity (%)", min_value=14.0, max_value=100.0, value=82.0, step=0.1)
    rainfall    = st.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=200.0, step=0.1)

st.markdown("---")

# ── Input Summary Table ────────────────────────────────────
st.subheader("📊 Your Input Summary")
input_data = {
    "Feature"   : ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "pH Level", "Rainfall"],
    "Value"     : [N, P, K, f"{temperature}°C", f"{humidity}%", ph, f"{rainfall}mm"],
    "Importance": ["🥉 10.4%", "4th 14.8%", "🥈 17.6%", "6th 7.8%", "🥇 20.5%", "7th 5.8%", "🏆 22.8%"]
}
st.dataframe(pd.DataFrame(input_data), use_container_width=True)

st.markdown("---")

# ── Predict Button ─────────────────────────────────────────
predict_btn = st.button("🌱 Recommend Crop", use_container_width=True)

if predict_btn:
    features = [[N, P, K, temperature, humidity, ph, rainfall]]
    pred     = model.predict(features)[0]

    crop = str(CROP_NAMES[int(pred)])

    st.markdown("---")
    st.subheader("✅ Prediction Result")
    st.success(f"🌾 Recommended Crop: **{crop.upper()}**")

    # Crop tips
    crop_tips = {
        "rice"       : "🌊 Rice needs high water — ensure proper irrigation.",
        "maize"      : "☀️ Maize grows best in warm, sunny conditions.",
        "chickpea"   : "🌡️ Chickpea prefers cool, dry weather.",
        "kidneybeans": "💧 Kidney beans need moderate rainfall.",
        "mango"      : "🥭 Mango loves tropical, humid climate.",
        "apple"      : "❄️ Apple needs cold winters for proper growth.",
        "banana"     : "🍌 Banana thrives in humid tropical regions.",
        "cotton"     : "☀️ Cotton needs dry, warm climate with good sunlight.",
        "coffee"     : "☕ Coffee grows best in cool, humid highlands.",
        "coconut"    : "🥥 Coconut loves coastal, humid regions.",
        "grapes"     : "🍇 Grapes need dry summers and cool winters.",
        "watermelon" : "🍉 Watermelon loves hot, dry weather.",
        "orange"     : "🍊 Orange needs subtropical, sunny climate.",
        "papaya"     : "🌴 Papaya grows in tropical, frost-free regions.",
        "pomegranate": "🌸 Pomegranate suits semi-arid, dry climate.",
        "jute"       : "🌿 Jute grows best in warm, humid conditions.",
        "lentil"     : "🌾 Lentil prefers cool, dry climate.",
        "blackgram"  : "🫘 Black gram needs warm, humid weather.",
        "mungbean"   : "🌱 Mung bean grows in warm, sunny conditions.",
        "mothbeans"  : "🌵 Moth beans suit dry, arid regions.",
        "muskmelon"  : "🍈 Musk melon needs hot, dry climate.",
        "pigeonpeas" : "🫘 Pigeon peas grow in semi-arid, tropical regions.",
    }
    if crop.lower() in crop_tips:
        st.info(crop_tips[crop.lower()])

    # ── Probability Chart ──────────────────────────────────
    st.markdown("#### 📊 Top 5 Crop Probabilities")
    proba = model.predict_proba(features)[0]

    crop_proba = dict(zip(range(len(CROP_NAMES)), proba))
    top5       = dict(sorted(crop_proba.items(), key=lambda x: -x[1])[:5])

    
    proba_df = pd.DataFrame({
        "Crop"       : [CROP_NAMES[int(i)].capitalize() for i in top5.keys()],
        "Probability": [round(v * 100, 2) for v in top5.values()]
    })
    st.dataframe(proba_df, use_container_width=True)
    st.bar_chart(proba_df.set_index("Crop"))

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About This App")
    st.markdown("""
    This app uses **Machine Learning** to recommend
    the best crop based on soil nutrients and climate conditions.

    ---
    **Model:** Random Forest Classifier
    **Accuracy:** 99.3%
    **Dataset:** 2200 samples, 22 crops

    **Features Used:**
    - Nitrogen, Phosphorus, Potassium
    - Temperature, Humidity
    - pH Level, Rainfall

    ---
    **Most Important Features:**
    1. 🏆 Rainfall (22.8%)
    2. 🥇 Humidity (20.5%)
    3. 🥈 Potassium (17.6%)
    4. 🥉 Phosphorus (14.8%)

    ---
    **Built by:** Vishnu Sharma
    **Tech Stack:** Python, Scikit-learn, SHAP, Streamlit
    """)

    st.markdown("---")
    st.markdown("### 🌱 22 Supported Crops")
    for c in ["Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas",
              "Moth Beans", "Mung Bean", "Black Gram", "Lentil", "Pomegranate",
              "Banana", "Mango", "Grapes", "Watermelon", "Musk Melon",
              "Apple", "Orange", "Papaya", "Coconut", "Cotton", "Jute", "Coffee"]:
        st.markdown(f"• {c}")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Made by Vishnu Sharma | B.Tech IT — JECRC Foundation, Jaipur</center>",
    unsafe_allow_html=True
)
