import streamlit as st
import time
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import gdown

st.set_page_config(
    page_title="AI Manufacturing Defect Detection",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color:#f6f8fb;}
.block-container {padding-top:2rem;}
h1,h2,h3 {color:#12355b;}
[data-testid="stSidebar"]{
    background-color:#0f172a;
}
[data-testid="stSidebar"] *{
    color:white;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "manufacturing_defect_cnn.keras"
FILE_ID = "1zBfYGyszK4dwIbCYDs4y_kYxJoAP_26J"

@st.cache_resource
def load_cnn():

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained CNN model... Please wait..."):
            gdown.download(
                id=FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    return load_model(MODEL_PATH)

model = load_cnn()

class_names = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches"
]

descriptions = {
    "Crazing":"Fine surface cracks.",
    "Inclusion":"Foreign material trapped in steel.",
    "Patches":"Irregular patch-like defects.",
    "Pitted Surface":"Small pits on the surface.",
    "Rolled-in Scale":"Oxide scale rolled into the surface.",
    "Scratches":"Linear surface scratches."
}

st.sidebar.title("🏭 Manufacturing Defect Detection")
st.sidebar.markdown("""
### Model Information
- **Model:** CNN
- **Framework:** TensorFlow / Keras
- **Dataset:** NEU-DET
- **Image Size:** 224×224
- **Validation Accuracy:** **83.3%**
""")

st.title("🔍 AI-Based Manufacturing Defect Detection")
st.write("Upload a steel surface image and let the trained CNN classify the manufacturing defect.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    c1,c2 = st.columns([1.1,1])

    with c1:
        st.subheader("Uploaded Image")
        st.image(img, use_container_width=True)

    start = time.time()

    img_resize = img.resize((224,224))
    arr = image.img_to_array(img_resize)
    arr = arr/255.0
    arr = np.expand_dims(arr,0)

    pred = model.predict(arr, verbose=0)

    idx = int(np.argmax(pred))
    confidence = float(np.max(pred))*100
    inference = time.time()-start

    with c2:
        st.subheader("Prediction")

        if confidence >= 90:
            st.success(f"✅ {class_names[idx]}")
        elif confidence >= 70:
            st.warning(class_names[idx])
        else:
            st.error(class_names[idx])

        a,b = st.columns(2)
        a.metric("Confidence", f"{confidence:.2f}%")
        b.metric("Inference Time", f"{inference:.3f} sec")

        st.progress(confidence/100)

        st.info(descriptions[class_names[idx]])

    st.divider()

    st.subheader("Prediction Probability Distribution")

    df = pd.DataFrame({
        "Defect": class_names,
        "Probability (%)": pred[0]*100
    }).set_index("Defect")

    st.bar_chart(df)

    with st.expander("Detailed Probabilities"):
        st.dataframe(df.style.format("{:.2f}"))

st.divider()

with st.expander("📘 About this Project"):
    st.markdown("""
### AI-Based Manufacturing Defect Detection

This application uses a Convolutional Neural Network (CNN) trained on the **NEU-DET** dataset to classify steel surface defects.

**Technologies**
- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Pillow

**Supported Classes**
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches
""")

st.markdown("---")
st.markdown(
"<center><h4>Developed by <b>Ojas Savkar</b></h4>"
"<p>Mechanical Engineering • AI • Deep Learning</p>"
"<p>TensorFlow | Keras | Streamlit</p></center>",
unsafe_allow_html=True
)
