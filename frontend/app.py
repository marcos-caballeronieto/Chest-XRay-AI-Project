import streamlit as st
import requests
from PIL import Image
import io

# 1. Page Configuration
st.set_page_config(
    page_title="Pneumonia Triage AI",
    page_icon="🫁",
    layout="centered"
)

# FastAPI Endpoint (Make sure your FastAPI server is running!)
API_URL = "http://127.0.0.1:8000/predict"

# 2. Header Section
st.title("🫁 Chest X-Ray Pneumonia Triage")
st.markdown("""
**Clinical Inference Engine:** DenseNet121 (Fine-Tuned at 448px)  
**Safety Protocol:** 3-Way Test-Time Augmentation (TTA) Consensus  
*Optimized for 0 False Negatives.*
""")
st.divider()

# 3. File Uploader
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Radiograph", use_container_width=True)

    # 4. Analyze Button
    if st.button("🧠 Analyze Radiograph", use_container_width=True):
        with st.spinner("Running 3-way geometric consensus (TTA)..."):
            try:
                # Prepare the file to send to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Make the POST request
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    st.subheader("📋 Triage Results")
                    
                    # 5. Display Diagnosis with conditional formatting
                    diagnosis = result["diagnosis"]
                    if diagnosis == "Pneumonia":
                        st.error(f"**DIAGNOSIS: {diagnosis.upper()} DETECTED**")
                    else:
                        st.success(f"**DIAGNOSIS: {diagnosis.upper()}**")
                    
                    # 6. Display Clinical Metrics
                    st.markdown(f"**TTA Consensus (Votes for Pneumonia):** {result['confidence_votes']}")
                    
                    # Expandable section for raw probabilities
                    with st.expander("View Raw Probabilities (Softmax)"):
                        probs = result["raw_probabilities"]
                        st.write(f"- **Original (448px):** {probs['original_448px']:.2%}")
                        st.write(f"- **Rotated (10°):** {probs['rotated_448px']:.2%}")
                        st.write(f"- **Zoomed (Crop):** {probs['zoomed_448px']:.2%}")
                        st.caption(result["clinical_metrics"])
                        
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the API. Is your FastAPI server running on port 8000?")