import streamlit as st
import numpy as np
from PIL import Image
import cv2

# --- Page Configuration ---
st.set_page_config(
    page_title="ImageLab Lite",
    layout="wide"
)

# --- App Title and Description ---
st.title("ImageLab Lite")
st.write("A simple image processing app by Chetan. Upload your photo and see the magic!")
st.write("---")

# --- Image Uploader ---
uploaded_file = st.file_uploader("Upload your image here:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Divide the page into two columns
    col1, col2 = st.columns(2)

    # Open and display the original image in the first column
    original_image = Image.open(uploaded_file)
    with col1:
        st.header("Original")
        st.image(original_image, caption='Your uploaded image.', use_container_width=True)

    # Convert the image for processing
    img_array = np.array(original_image)

    # Convert to grayscale using OpenCV
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Display the processed image in the second column
    with col2:
        st.header("Grayscale")
        st.image(gray_image, caption='The processed grayscale image.', use_container_width=True)
else:
    st.info("Please upload an image file to begin.")
