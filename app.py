import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="ImageLab Pro",
    layout="wide"
)

# Function to convert image for download
def convert_image_to_bytes(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# --- App Title and Sidebar ---
st.title("ImageLab Pro")
st.write("An advanced image processing app by Chetan. Upload a photo and use the controls to edit!")
st.write("---")

# Sidebar for controls
st.sidebar.header("Image Controls")
uploaded_file = st.sidebar.file_uploader("Upload your image:", type=["jpg", "png", "jpeg"])

# --- Main App Logic ---
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(original_image)
    # Convert RGB to BGR for OpenCV
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Display the original image
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original")
        st.image(original_image, caption='Your uploaded image.', use_container_width=True)

    # --- Add controls to the sidebar ---
    st.sidebar.header("Editing Tools")
    
    # Sliders for adjustments
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", -100, 100, 0)
    saturation = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, 0.1)
    
    # Effect selection
    effect = st.sidebar.selectbox(
        "Apply a special effect",
        ["None", "Sharpen", "Grayscale", "Pencil Sketch", "Sepia"]
    )

    # --- Image Processing Pipeline ---
    # 1. Brightness and Contrast
    processed_img = cv2.convertScaleAbs(img_array_bgr, alpha=1 + contrast / 100.0, beta=brightness)

    # 2. Saturation
    if saturation != 1.0:
        hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, saturation)
        s = np.clip(s, 0, 255)
        final_hsv = cv2.merge([h, s, v])
        processed_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # 3. Special Effects
    if effect == "Sharpen":
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        processed_img = cv2.filter2D(processed_img, -1, kernel)
        caption_text = 'Sharpened'

    elif effect == "Grayscale":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        caption_text = 'Grayscale effect.'
        
    elif effect == "Pencil Sketch":
        gray_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        inv_gray_image = 255 - gray_image
        blur_image = cv2.GaussianBlur(inv_gray_image, (21, 21), 0)
        inv_blur_image = 255 - blur_image
        processed_img = cv2.divide(gray_image, inv_blur_image, scale=256.0)
        caption_text = 'Pencil Sketch effect.'

    elif effect == "Sepia":
        # Convert back to RGB for Sepia kernel which expects RGB order
        rgb_img_for_sepia = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        kernel = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]])
        sepia_img_array = cv2.transform(rgb_img_for_sepia, kernel.T) # Use transpose for correct multiplication
        processed_img_bgr = np.clip(sepia_img_array, 0, 255).astype(np.uint8)
        processed_img = cv2.cvtColor(processed_img_bgr, cv2.COLOR_RGB2BGR) # convert back to BGR for consistency
        caption_text = 'Sepia effect.'
    else: # None
        caption_text = 'Adjusted Brightness, Contrast, & Saturation.'

    # Display the processed image in the second column
    with col2:
        st.header("Processed")
        # Convert final image back to RGB for displaying in Streamlit
        if len(processed_img.shape) == 3: # Check if it's a color image
             display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        else: # It's grayscale
            display_img = processed_img
        st.image(display_img, caption=caption_text, use_container_width=True)

        # Download button
        st.download_button(
           label="Download Processed Image",
           data=convert_image_to_bytes(processed_img),
           file_name=f"processed_{effect.lower().replace(' ', '_')}.png",
           mime="image/png"
        )

else:
    st.info("Upload an image using the sidebar to start editing.")
