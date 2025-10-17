import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import time
import imageio

st.set_page_config(
    page_title="ImageLab Pro", # Changed from ImageLab MotionStudio
    page_icon="🎨", # Changed icon
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

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
    spread = 2.0 - strength / 100.0 if strength > 0 else 0.01 
    kernel_x = cv2.getGaussianKernel(cols, cols / spread)
    kernel_y = cv2.getGaussianKernel(rows, rows / spread)
    kernel = kernel_y * kernel_x.T
    
    mask = (kernel / np.max(kernel)) 
    
    output = np.copy(img).astype(np.float32) / 255.0
    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask
    
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output

def generate_rain_frame(img, frame_num, num_frames, density=0.02, speed=1):
    h, w, _ = img.shape
    rain_mask = np.zeros_like(img, dtype=np.uint8)
    
    num_drops = int(w * h * density)
    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y_start = (np.random.randint(-h, h) + frame_num * speed * 20) % (2 * h) - h
        y_end = y_start + h // 20
        
        cv2.line(rain_mask, (x, y_start), (x, y_end), (200, 200, 200), 1)
        
    rain_mask = cv2.GaussianBlur(rain_mask, (3, 3), 0)
    
    animated_frame = cv2.addWeighted(img, 0.7, rain_mask, 0.3, 0)
    return animated_frame

def generate_snow_frame(img, frame_num, num_frames, density=0.01, flake_size=2, speed=0.5):
    h, w, _ = img.shape
    snow_mask = np.zeros_like(img, dtype=np.uint8)
    
    num_flakes = int(w * h * density)
    for i in range(num_flakes):
        offset = (np.random.randint(0, h * 2) + frame_num * speed * 10) % (h * 2) 
        x = i % w 
        y = (i // w * 5 + offset) % h
        cv2.circle(snow_mask, (x, y), flake_size, (255, 255, 255), -1)
        
    snow_mask = cv2.GaussianBlur(snow_mask, (3, 3), 0)
    
    animated_frame = cv2.addWeighted(img, 0.85, snow_mask, 0.15, 0)
    return animated_frame

def generate_animated_gif(img_bgr, animation_type, num_frames=20, duration_per_frame=0.1):
    frames = []
    h, w, _ = img_bgr.shape
    
    img_rgb_base = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    for i in range(num_frames):
        if animation_type == "Rain":
            frame = generate_rain_frame(img_rgb_base, i, num_frames)
        elif animation_type == "Snow":
            frame = generate_snow_frame(img_rgb_base, i, num_frames)
        else:
            frame = img_rgb_base
        frames.append(frame)

    gif_bytes_io = io.BytesIO()
    imageio.mimsave(gif_bytes_io, frames, format='gif', duration=duration_per_frame)
    gif_bytes_io.seek(0)
    return gif_bytes_io.getvalue()

st.markdown('<div class="header"><h1>🎨 ImageLab Pro</h1></div>', unsafe_allow_html=True) # Updated title and icon
st.markdown('<p class="subheader">Crafted by <strong>Chetan</strong>. Your professional studio for advanced digital image processing.</p>', unsafe_allow_html=True) # Updated subheader

st.sidebar.header("🛠️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload your image to begin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Loading image..."):
        original_image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(original_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        time.sleep(0.5)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<h3 class="image-title">Original</h3>', unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True, channels="RGB")
        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    
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
        
    with st.sidebar.expander("🌟 Static Effects & Filters"):
        sharpness = st.slider("Sharpness", 0, 100, 0, key='sharpness_slider')
        blur = st.slider("Blur", 0, 100, 0, key='blur_slider')
        vignette_strength = st.slider("Vignette", 0, 100, 0, key='vignette_slider')
        effect = st.selectbox("Apply a filter", ["None", "Grayscale", "Pencil Sketch", "Sepia"], key='effect_selector')

    st.sidebar.markdown("---")
    with st.sidebar.expander("✨ Animated Effects"):
        animated_effect = st.selectbox(
            "Add animated weather to your photo!",
            ["None", "Rain", "Snow"],
            key='animated_effect_selector'
        )
        animation_frames = st.slider("Animation Frames", 10, 50, 20, key='anim_frames_slider')
        animation_speed = st.slider("Animation Speed", 0.05, 0.2, 0.1, 0.01, key='anim_speed_slider')


    st.markdown('<div class="processing-status">Processing Image...</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    processed_img = img_bgr.copy()
    current_step_list = [0] 
    processing_steps = 7 if animated_effect == "None" else 8

    def update_progress_fix():
        current_step_list[0] += 1
        progress_bar.progress(current_step_list[0] / processing_steps)
        time.sleep(0.05)

    processed_img = cv2.convertScaleAbs(processed_img, alpha=1 + contrast / 100.0, beta=brightness)
    update_progress_fix()

    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed_img = cv2.LUT(processed_img, table)
    update_progress_fix()

    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(cv2.multiply(s, saturation), 0, 255).astype(np.uint8)
    h = ((h.astype(int) + hue) % 180).astype(np.uint8)
    processed_img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    b, g, r = cv2.split(processed_img)
    r = np.clip(cv2.add(r, red_balance), 0, 255).astype(np.uint8)
    g = np.clip(cv2.add(g, green_balance), 0, 255).astype(np.uint8)
    b = np.clip(cv2.subtract(b, temp * 0.5), 0, 255).astype(np.uint8) 
    processed_img = cv2.merge([b, g, r])
    update_progress_fix()

    if sharpness > 0:
        kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 10.0, -1], [-1, -1, -1]])
        processed_img = cv2.filter2D(processed_img, -1, kernel)
    update_progress_fix()

    if blur > 0:
        k_size = (blur * 2) + 1
        processed_img = cv2.GaussianBlur(processed_img, (k_size, k_size), 0)
    update_progress_fix()

    if vignette_strength > 0:
        processed_img = apply_vignette(processed_img, vignette_strength)
    update_progress_fix()

    if effect == "Grayscale":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    elif effect == "Pencil Sketch":
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blur_fx = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        processed_img = cv2.divide(gray, inv_blur_fx, scale=256.0)
    elif effect == "Sepia":
        sepia_kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(processed_img, sepia_kernel)
        processed_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    update_progress_fix()
    
    final_output_image = processed_img

    animated_gif_bytes = None
    if animated_effect != "None":
        st.markdown(f'<div class="processing-status">Generating {animated_effect} Animation...</div>', unsafe_allow_html=True)
        animated_gif_bytes = generate_animated_gif(final_output_image, animated_effect, 
                                                    num_frames=animation_frames, duration_per_frame=animation_speed)
        update_progress_fix()


    progress_bar.empty()
    st.markdown('<div class="processing-status-complete">Processing Complete!</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="image-title">Processed</h3>', unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        if animated_gif_bytes:
            st.image(animated_gif_bytes, use_container_width=True, caption=f'{animated_effect} Effect Applied.')
        else:
            st.image(final_output_image, channels="BGR" if len(final_output_image.shape)==3 else "GRAY", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
        
    st.sidebar.markdown("---")
    if animated_gif_bytes:
        st.sidebar.download_button(
            label="⬇️ Download Animated GIF",
            data=animated_gif_bytes,
            file_name=f"ImageLab_{animated_effect.lower()}_animation.gif",
            mime="image/gif",
            use_container_width=True
        )
    else:
        st.sidebar.download_button(
            label="📥 Download Edited Image",
            data=convert_image_to_bytes(final_output_image),
            file_name="ImageLab_edited_photo.png",
            mime="image/png",
            use_container_width=True
        )

else:
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="landing-message">Unleash Your Creativity</h2>', unsafe_allow_html=True)
    st.image("https://res.cloudinary.com/demo/image/upload/v1605389650/sample_image.jpg", caption="Photo by Cloudinary (creative abstract art)", width=550)
    st.markdown('<p class="upload-prompt">Upload an image using the sidebar to begin your transformation!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
