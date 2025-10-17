import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import time # For simulating processing time with animations

# --- Page Configuration ---
st.set_page_config(
    page_title="ImageLab Pro-Animator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Injection ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- Helper Functions ---
def convert_image_to_bytes(img, format="PNG"):
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    
    buf = io.BytesIO()
    img.save(buf, format=format)
    byte_im = buf.getvalue()
    return byte_im

def apply_vignette(img, strength):
    if strength == 0:
        return img
    rows, cols = img.shape[:2]
    # Reduce spread for a more noticeable vignette with lower strength
    spread = 2.0 - strength / 100.0 if strength > 0 else 0.01 
    kernel_x = cv2.getGaussianKernel(cols, cols / spread)
    kernel_y = cv2.getGaussianKernel(rows, rows / spread)
    kernel = kernel_y * kernel_x.T
    
    # Normalize kernel to 0-1 range and make it darker towards edges
    mask = (kernel / np.max(kernel)) 
    
    # Apply to each channel
    output = np.copy(img).astype(np.float32) / 255.0 # Normalize to 0-1
    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask
    
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output

# --- Header Section ---
st.markdown('<div class="header"><h1>✨ ImageLab Pro-Animator</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Crafted by <strong>Chetan</strong>. Your ultimate studio for breathtaking photo transformations!</p>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("🛠️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload your image to begin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show a spinner while the image is being loaded
    with st.spinner("Loading image..."):
        original_image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(original_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        time.sleep(0.5) # Simulate brief loading

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<h3 class="image-title">Original Image</h3>', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True, channels="RGB")

    st.sidebar.markdown("---") # Visual separator
    
    with st.sidebar.expander("⚪ Basic Adjustments", expanded=True):
        brightness = st.slider("Brightness", -100, 100, 0, key='brightness_slider')
        contrast = st.slider("Contrast", -100, 100, 0, key='contrast_slider')
        gamma = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1, key='gamma_slider')

    with st.sidebar.expander("🎨 Color & Tone"):
        saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.1, key='saturation_slider')
        hue = st.slider("Hue Shift", -180, 180, 0, key='hue_slider')
        temp = st.slider("Temperature", -100, 100, 0, key='temp_slider')
        st.markdown("<strong>Color Balance</strong>", unsafe_allow_html=True)
        red_balance = st.slider("Red", -50, 50, 0, key='red_balance_slider')
        green_balance = st.slider("Green", -50, 50, 0, key='green_balance_slider')
        blue_balance = st.slider("Blue", -50, 50, 0, key='blue_balance_slider')
        
    with st.sidebar.expander("🌟 Effects & Filters"):
        sharpness = st.slider("Sharpness", 0, 100, 0, key='sharpness_slider')
        blur = st.slider("Blur", 0, 100, 0, key='blur_slider')
        vignette_strength = st.slider("Vignette", 0, 100, 0, key='vignette_slider')
        effect = st.selectbox("Apply a special filter", ["None", "Grayscale", "Pencil Sketch", "Sepia"], key='effect_selector')

    # --- Image Processing Pipeline ---
    # Show a progress bar while processing
    st.markdown('<div class="processing-status">Processing Image...</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    processed_img = img_bgr.copy()
    processing_steps = 7 # Number of major processing steps
    current_step = 0

    def update_progress():
        nonlocal current_step
        current_step += 1
        progress_bar.progress(current_step / processing_steps)
        time.sleep(0.05) # Small delay for animation effect

    # 1. Brightness & Contrast
    processed_img = cv2.convertScaleAbs(processed_img, alpha=1 + contrast / 100.0, beta=brightness)
    update_progress()

    # 2. Gamma
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed_img = cv2.LUT(processed_img, table)
    update_progress()

    # 3. Saturation & Hue & Temperature
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(cv2.multiply(s, saturation), 0, 255).astype(np.uint8)
    h = ((h.astype(int) + hue) % 180).astype(np.uint8)
    processed_img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    b, g, r = cv2.split(processed_img)
    r = np.clip(cv2.add(r, red_balance), 0, 255).astype(np.uint8)
    g = np.clip(cv2.add(g, green_balance), 0, 255).astype(np.uint8)
    b = np.clip(cv2.add(b, blue_balance), 0, 255).astype(np.uint8) 
    r = np.clip(cv2.add(r, temp * 0.5), 0, 255).astype(np.uint8) 
    b = np.clip(cv2.subtract(b, temp * 0.5), 0, 255).astype(np.uint8) 
    processed_img = cv2.merge([b, g, r])
    update_progress()

    # 4. Sharpness
    if sharpness > 0:
        kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 10.0, -1], [-1, -1, -1]])
        processed_img = cv2.filter2D(processed_img, -1, kernel)
    update_progress()

    # 5. Blur
    if blur > 0:
        k_size = (blur * 2) + 1 # Ensure odd kernel size
        processed_img = cv2.GaussianBlur(processed_img, (k_size, k_size), 0)
    update_progress()

    # 6. Vignette
    if vignette_strength > 0:
        processed_img = apply_vignette(processed_img, vignette_strength)
    update_progress()

    # 7. Final Effects
    if effect == "Grayscale":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    elif effect == "Pencil Sketch":
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blur_fx = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        processed_img = cv2.divide(gray, 255 - blur_fx, scale=256.0)
    elif effect == "Sepia":
        sepia_kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(processed_img, sepia_kernel)
        processed_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    update_progress()
    
    # Hide progress bar after completion (optional, it disappears automatically when finished)
    progress_bar.empty()
    st.markdown('<div class="processing-status-complete">Processing Complete!</div>', unsafe_allow_html=True)


    # --- Display Processed Image ---
    with col2:
        st.markdown('<h3 class="image-title">Processed Image</h3>', unsafe_allow_html=True)
        st.image(processed_img, channels="BGR" if len(processed_img.shape)==3 else "GRAY", use_container_width=True)
        
        st.sidebar.markdown("---")
        st.sidebar.download_button(
           label="📥 Download Edited Image",
           data=convert_image_to_bytes(processed_img),
           file_name="ImageLab_edited_photo.png",
           mime="image/png",
           use_container_width=True
        )

else:
    # --- Animated Landing Page ---
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="landing-message">Unleash Your Creativity</h2>', unsafe_allow_html=True)
    st.image("https://res.cloudinary.com/demo/image/upload/v1605389650/sample_image.jpg", caption="Photo by Cloudinary (creative abstract art)", width=550)
    st.markdown('<p class="upload-prompt">Upload an image using the sidebar to begin your transformation!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
