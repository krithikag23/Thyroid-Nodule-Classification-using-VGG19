import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('thyroid_nodule_classifier.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to the size expected by the model
    image = img_to_array(image) / 255.0  # Convert to array and normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
def preprocess_image(image):
    # Load and preprocess the image
    image = image.resize((150, 150))
    img_array = img_to_array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
# Custom CSS for background image and translucent glass block
st.markdown("""
    <style>
    body {
        background-image: url("https://static.vecteezy.com/system/resources/thumbnails/004/987/898/small_2x/doctor-in-medical-lab-coat-with-a-stethoscope-doctor-in-hospital-background-with-copy-space-low-poly-wireframe-vector.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    .glass {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    .center-title {
        text-align: center;
        color: white;
    }
    .bold-text {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://static.vecteezy.com/system/resources/previews/004/987/905/non_2x/doctor-in-hospital-background-with-copy-space-free-vector.jpg");
background-size: cover;
}
</style>
'''

# Centered title in a glass block

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<div class='glass'><h1 class='center-title'>Thyroid nodule cancer detector</h1>", unsafe_allow_html=True)

st.write("<div class='center-title'>Upload an image to see if it's benign or malignant", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns(2)

    # Display the image in the first column
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)

    # Extract a specific part of the file path if needed (e.g., split by '/')
    # In this example, we'll just display the file name
    pred = uploaded_file.name.split('/')[1] if '/' in uploaded_file.name else uploaded_file.name

    # Display the result in the second column
    with col2:
        if prediction[0][0] < 0.5:
            result_text = f"The model predicts: <span class='bold-text'>Benign</span> for the image '{pred}'"
        else:
            result_text = f"The model predicts: <span class='bold-text'>Malignant</span> for the image '{pred}'"
        st.markdown(result_text, unsafe_allow_html=True)

if st.button("About"):
    st.write("""
    This is a simple web app to classify whether a photo is benign or malignant.
    The model is trained using a machine learning algorithm.
    """)
st.markdown("</div>", unsafe_allow_html=True)
