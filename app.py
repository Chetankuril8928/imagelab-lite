import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io

st.set_page_config(
    page_title="ImageLab Studio",
    page_icon="🎨",
    layout="wide"
)

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
    kernel_x = cv2.getGaussianKernel(cols, cols / (2.0 - strength / 100.0))
    kernel_y = cv2.getGaussianKernel(rows, rows / (2.0 - strength / 100.0))
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output = np.copy(img)
    
    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask
    return output.astype(np.uint8)

st.title("🎨 ImageLab Studio")
st.markdown("### A comprehensive image editor by **Chetan**. Your all-in-one solution for stunning photos.")

st.sidebar.header("🛠️ Image Controls")
uploaded_file = st.sidebar.file_uploader("Upload your image to begin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(original_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("Original")
        st.image(original_image, use_container_width=True)

    st.sidebar.divider()
    
    with st.sidebar.expander("⚪ Basic Adjustments", expanded=True):
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", -100, 100, 0)
        gamma = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)

    with st.sidebar.expander("🎨 Color & Tone"):
        saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.1)
        hue = st.slider("Hue Shift", -180, 180, 0)
        temp = st.slider("Temperature", -100, 100, 0)
        st.subheader("Color Balance")
        red_balance = st.slider("Red", -50, 50, 0)
        green_balance = st.slider("Green", -50, 50, 0)
        blue_balance = st.slider("Blue", -50, 50, 0)
        
    with st.sidebar.expander("🌟 Effects & Filters"):
        sharpness = st.slider("Sharpness", 0, 100, 0)
        blur = st.slider("Blur", 0, 100, 0)
        vignette_strength = st.slider("Vignette", 0, 100, 0)
        effect = st.selectbox("Apply a filter", ["None", "Grayscale", "Pencil Sketch", "Sepia"])

    processed_img = img_bgr.copy()

    processed_img = cv2.convertScaleAbs(processed_img, alpha=1 + contrast / 100.0, beta=brightness)

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    processed_img = cv2.LUT(processed_img, table)

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

    if sharpness > 0:
        kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 10.0, -1], [-1, -1, -1]])
        processed_img = cv2.filter2D(processed_img, -1, kernel)

    if blur > 0:
        k_size = (blur * 2) + 1
        processed_img = cv2.GaussianBlur(processed_img, (k_size, k_size), 0)

    if vignette_strength > 0:
        processed_img = apply_vignette(processed_img, vignette_strength)

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

    with col2:
        st.header("Processed")
        st.image(processed_img, channels="BGR" if len(processed_img.shape)==3 else "GRAY", use_container_width=True)
        
        st.sidebar.divider()
        st.sidebar.subheader("📥 Download")
        st.sidebar.download_button(
           label="Save Processed Image",
           data=convert_image_to_bytes(processed_img),
           file_name="processed_image.png",
           mime="image/png",
           use_container_width=True
        )

else:
    new_image_url = "https://images.unsplash.com/photo-1569336415962-a4bd9f69cd83"
    st.image(new_image_url, caption="Photo by Brizmaker on Unsplash", width=550)
    st.markdown("<h2 style='text-align: center;'>Upload an image to start your creative journey!</h2>", unsafe_allow_html=True)
