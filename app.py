import streamlit as st
from ultralytics import YOLO
import os
import tempfile
from PIL import Image

# Set page config as the very first call
st.set_page_config(
    page_title="Rice Leaf Disease Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO('https://www.dropbox.com/scl/fi/uomn3tirx8add2s4u1v80/best.onnx?rlkey=hnu6pxbv517cqf1epefm7au3z&st=nntyhdns&dl=1', task='classify')

model = load_model()

def set_bg_gradient():
    """Sets a gradient background color."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(to bottom right, #283048, #859398);
            color: white;
        }}
        h1 {{
            color: #FFD700;
            text-align: center;
            font-family: 'Arial', sans-serif;
            margin-bottom: 0;
        }}
        .stTextInput > div > label {{
            color: #FFD700;
        }}
        .stImage {{
            border-radius: 10px;
        }}
        .stFileUploader label {{
            color: white;
        }}
        .stMarkdown p {{
            color: white;
            font-size: 1.2em;
        }}
        .st-success {{
            background-color: #1b1e23;
            border-color: #4caf50;
        }}
        .st-success p {{
            color: #4caf50;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the gradient background
set_bg_gradient()

# Streamlit interface setup
st.markdown("<h1>Rice Leaf Disease Classification</h1>", unsafe_allow_html=True)

# Function to handle image file upload and prediction
def handle_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            # Predict using the model with the image file path
            results = model(tmp_file_path, imgsz=224)
            os.unlink(tmp_file_path)  # Delete the temp file after prediction
            return results
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Sidebar for uploading images
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
    st.image("https://www.dropbox.com/scl/fi/9t0701yh6ckl7lrx4s6e0/logo.jpeg?rlkey=sbfgd12fgut0otwy6o86adol7&st=c8qbq2vy&dl=1", width=200)

# Process the uploaded image
if uploaded_file is not None:
    results = handle_uploaded_file(uploaded_file)
    if isinstance(results, str):  # Check if results is the error message
        st.error(results)
    else:
        col1, col2 = st.columns([3, 2])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            if results:
                top1_index = results[0].probs.top1
                top1_conf = results[0].probs.top1conf * 100
                label_names = results[0].names

                top1_label = label_names[top1_index]

                st.subheader("Prediction Results:")
                st.success(f"**Prediction:** {top1_label})")

