import urllib.request
import streamlit as st
from fastai.vision.all import *
import pathlib
import platform

MODEL_URL = "https://github.com/Prathamesh-2506/03-ai-eye-disease-detection/releases/download/v1.0/eye_disease_model.pkl"
MODEL_PATH = "eye_disease_model.pkl"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
with st.spinner("Downloading AI model..."):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Windows path compatibility
system = platform.system()
if system == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Load model
learn = load_learner(MODEL_PATH)

# Disease knowledge base
DISEASE_INFO = {
    "Cataracts": {
        "details": "Clouding of the eye lens that causes blurry or dim vision.",
        "symptoms": [
            "Blurred vision",
            "Difficulty seeing at night",
            "Sensitivity to light",
            "Faded colors"
        ],
        "precautions": [
            "Wear UV-protective sunglasses",
            "Control diabetes and blood sugar",
            "Avoid smoking",
            "Regular eye checkups"
        ],
        "doctor": "Consult an ophthalmologist for slit-lamp examination and cataract surgery guidance if severe."
    },
    "Glaucoma": {
        "details": "A condition that damages the optic nerve, often due to increased eye pressure.",
        "symptoms": [
            "Eye pain",
            "Tunnel vision",
            "Halos around lights",
            "Vision loss"
        ],
        "precautions": [
            "Regular eye pressure screening",
            "Use prescribed eye drops on time",
            "Avoid self-medication",
            "Maintain healthy blood pressure"
        ],
        "doctor": "Visit an eye specialist immediately for intraocular pressure testing."
    },
    "Uveitis": {
        "details": "Inflammation of the middle layer of the eye that may affect vision.",
        "symptoms": [
            "Red eyes",
            "Eye pain",
            "Floaters",
            "Blurred vision"
        ],
        "precautions": [
            "Protect eyes from dust",
            "Treat infections early",
            "Avoid rubbing eyes",
            "Follow anti-inflammatory treatment"
        ],
        "doctor": "Consult an ophthalmologist quickly to prevent retinal complications."
    },
    "Crossed_Eyes": {
        "details": "Misalignment of the eyes where they do not look in the same direction.",
        "symptoms": [
            "Double vision",
            "Eye strain",
            "Head tilt",
            "Poor depth perception"
        ],
        "precautions": [
            "Eye exercises if prescribed",
            "Use corrective glasses",
            "Early pediatric eye checkups",
            "Limit prolonged screen strain"
        ],
        "doctor": "Consult an eye specialist for vision therapy or alignment correction."
    },
    "Bulging eyes": {
        "details": "Protrusion of one or both eyes, often linked to thyroid-related issues.",
        "symptoms": [
            "Dry eyes",
            "Eye protrusion",
            "Difficulty closing eyes",
            "Double vision"
        ],
        "precautions": [
            "Check thyroid hormone levels",
            "Use lubricating eye drops",
            "Wear protective glasses",
            "Sleep with head elevated"
        ],
        "doctor": "Consult both an ophthalmologist and endocrinologist for thyroid eye disease evaluation."
    }
}

st.set_page_config(
    page_title="AI Eye Disease Detection",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ AI-Powered Eye Disease Detection")
st.markdown("Upload an eye image to predict the disease, symptoms, precautions, and medical guidance.")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Eye Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            pred, pred_idx, probs = learn.predict(img)

        confidence = float(probs[pred_idx]) * 100
        disease = str(pred)

        st.success(f"Predicted Disease: {disease}")
        st.progress(int(confidence))
        st.write(f"Confidence Score: {confidence:.2f}%")

    info = DISEASE_INFO.get(disease)

    if info:
        st.subheader("📖 Disease Details")
        st.info(info["details"])

        st.subheader("🩺 Symptoms")
        for symptom in info["symptoms"]:
            st.write(f"• {symptom}")

        st.subheader("🛡️ Precautions")
        for precaution in info["precautions"]:
            st.write(f"• {precaution}")

        st.subheader("👨‍⚕️ Doctor Suggestion")
        st.warning(info["doctor"])

    st.caption("This AI prediction is for educational purposes only and should not replace professional medical diagnosis.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 14px;'>© 2026 Prathamesh Shelke. All rights reserved.</div>", unsafe_allow_html=True)
