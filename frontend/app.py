import streamlit as st
import requests
from PIL import Image
import io
import base64
import streamlit.components.v1 as components

# 1. Page Configuration (Wide layout for medical dashboard)
st.set_page_config(
    page_title="Pneumonia Triage AI",
    page_icon="🫁",
    layout="wide"
)

# Inject CSS to prevent horizontal wiggle
st.markdown("""
    <style>
        /* Prevent horizontal overflow that causes layout wiggling with layout="wide" */
        html, body, [data-testid="stAppViewContainer"], .main, .block-container {
            overflow-x: hidden !important;
        }
        /* Hide the inner Streamlit scrollbar to prevent the "Double Scrollbar" UX bug in HF Spaces */
        ::-webkit-scrollbar {
            width: 0px;
            background: transparent; /* make scrollbar invisible */
        }
    </style>
""", unsafe_allow_html=True)

# 2. The Correct API URL (Cloud deployment)
API_URL = "https://marcoscaballero27-pneumonia-triage-api.hf.space/predict"
#API_URL = "http://127.0.0.1:8000/predict"

# 3. Premium Header Section
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="font-size: 3rem; font-weight: 800; background: -webkit-linear-gradient(#f8fafc, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🫁 Chest X-Ray Pneumonia Triage
    </h1>
    <p style="font-size: 1.1rem; color: #94a3b8;">
        <strong>Clinical Inference Engine:</strong> DenseNet121 (Fine-Tuned at 448px)<br>
        <strong>Safety Protocol:</strong> 3-Way Test-Time Augmentation (TTA) Consensus<br>
        <span style="color: #60a5fa; font-style: italic;">Optimized for 0 False Negatives.</span>
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# 4. CLINICAL LIABILITY DISCLAIMER 
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.error("""
    **⚠️ STRICTLY FOR ACADEMIC/PORTFOLIO PURPOSES.**
    
    This application is a machine learning experiment. It is **NOT** an FDA-approved medical device. 
    It must not be used for actual patient diagnosis, clinical triage, or treatment planning. 
    """)
    
    agree = st.checkbox("I understand and agree that this is not a medical tool.")
    if agree:
        st.session_state.disclaimer_accepted = True
        st.rerun()

else:
    # Add this explicit OOD warning (Restricted to AP only)
    st.warning("""
    ** Note on Image Types:** This model is strictly trained on **anterior-posterior (AP)** chest radiographs. 
    Uploading non-chest X-ray images (e.g., animals, everyday objects, or other body parts) will result in highly confident, but entirely nonsensical predictions.
    """, icon="⚠️")
    
    uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Side-by-side layout
        col_image, col_results = st.columns([1, 1])
        
        with col_image:
            st.subheader("Radiograph")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col_results:
            st.subheader("AI Analysis")
            
            if st.button("🧠 Run Triage Protocol", use_container_width=True):
                with st.spinner("Generating 448px crops and running geometric consensus (TTA)..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(API_URL, files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.divider()
                            
                            diagnosis = result["diagnosis"]
                            if diagnosis == "Pneumonia":
                                st.error(f"### 🚨 DIAGNOSIS: {diagnosis.upper()} DETECTED")
                            else:
                                st.success(f"### ✅ DIAGNOSIS: {diagnosis.upper()}")
                            
                            st.markdown(f"**TTA Consensus (Votes for Pneumonia):** `{result['confidence_votes']}`")
                            st.caption(result["clinical_metrics"])
                            
                            with st.expander("🔍 View Raw Probabilities (Softmax)", expanded=True):
                                probs = result["raw_probabilities"]
                                st.write(f"- **Original (448px):** {probs['original_448px']:.2%}")
                                st.write(f"- **Rotated (10°):** {probs['rotated_448px']:.2%}")
                                st.write(f"- **Zoomed (Crop):** {probs['zoomed_448px']:.2%}")
                                
                            if "gradcam_base64" in result and result["gradcam_base64"]:
                                st.divider()
                                st.subheader("Grad-CAM Attention Map")
                                gradcam_bytes = base64.b64decode(result["gradcam_base64"])
                                gradcam_image = Image.open(io.BytesIO(gradcam_bytes))
                                st.image(gradcam_image, caption="Red regions indicate high AI feature importance.", use_container_width=True)
                                st.info("ℹ️ **Future Roadmap Notice:** The heatmap may highlight peripheral areas like shoulders or arms, this is a known 'shortcut learning' effect. As documented in our V2 Roadmap, future versions will implement a Lung Segmentation Pipeline to force the model to focus exclusively on pulmonary parenchyma textures.")
                                
                        else:
                            st.error(f"API Error: {response.status_code} - {response.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("🚨 Could not connect to the Inference Engine. Ensure the API Space is awake.")
