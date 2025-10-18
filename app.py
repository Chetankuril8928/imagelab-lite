import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import time
import imageio

st.set_page_config(
    page_title="ImageLab Pro",
    page_icon="🎬",
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

def adjust_highlights_shadows(img, highlights=0, shadows=0):
    if highlights == 0 and shadows == 0:
        return img
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    if shadows != 0:
        shadow_mask = (L < 128).astype(np.float32) * (shadows / 100.0)
        L = np.clip(L + L * shadow_mask, 0, 255).astype(np.uint8)

    if highlights != 0:
        highlight_mask = (L > 128).astype(np.float32) * (highlights / 100.0)
        L = np.clip(L + L * highlight_mask, 0, 255).astype(np.uint8)
        
    final_lab = cv2.merge([L, A, B])
    return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

def add_film_grain(img, intensity=0):
    if intensity == 0:
        return img
    
    h, w, c = img.shape
    grain = np.random.normal(0, intensity * 2, (h, w, c)).astype(np.uint8)
    
    grained_img = cv2.addWeighted(img, 1.0, grain, 0.2, 0)
    return grained_img

def adjust_hsl(img, h_shift=0, s_mult=1.0, l_mult=1.0):
    if h_shift == 0 and s_mult == 1.0 and l_mult == 1.0:
        return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if h_shift != 0:
        h = ((h.astype(int) + h_shift) % 180).astype(np.uint8)
    
    s = np.clip(s * s_mult, 0, 255).astype(np.uint8)
    v = np.clip(v * l_mult, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

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

st.markdown('<div class="header"><h1>🎬 ImageLab Pro</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Crafted by <strong>Chetan</strong>. The definitive studio for dynamic photo editing.</p>', unsafe_allow_html=True)

st.sidebar.header("🛠️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload your image to begin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if 'current_file_id' not in st.session_state or uploaded_file.id != st.session_state.current_file_id:
        st.session_state.current_file_id = uploaded_file.id
        st.session_state.processing_complete = False

    if not st.session_state.processing_complete:
        st.markdown("""
        <div class="loading-container">
            <div class="spinner"></div>
            <p class="loading-text">Loading image...</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1)
        st.empty()
    
    original_image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(original_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<h3 class="image-title">Original</h3>', unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True, channels="RGB")
        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    with st.sidebar.expander("🎥 Cinematic Controls", expanded=True):
        exposure = st.slider("Exposure", -50, 50, 0)
        highlights = st.slider("Highlights", -50, 50, 0)
        shadows = st.slider("Shadows", -50, 50, 0)
        hsl_hue = st.slider("HSL Hue Shift", -180, 180, 0)
        hsl_sat = st.slider("HSL Saturation", 0.0, 3.0, 1.0, 0.1)
        hsl_light = st.slider("HSL Lightness", 0.0, 2.0, 1.0, 0.1)

    with st.sidebar.expander("⚪ Basic Adjustments"):
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", -100, 100, 0)
        gamma = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)

    with st.sidebar.expander("🎨 Global Color & Tone"):
        saturation = st.slider("Saturation (Global)", 0.0, 3.0, 1.0, 0.1)
        hue = st.slider("Hue Shift (Global)", -180, 180, 0)
        temp = st.slider("Temperature", -100, 100, 0)
        red_balance = st.slider("Red", -50, 50, 0)
        green_balance = st.slider("Green", -50, 50, 0)
        blue_balance = st.slider("Blue", -50, 50, 0)
        
    with st.sidebar.expander("🌟 Static Effects & Filters"):
        sharpness = st.slider("Sharpness", 0, 100, 0)
        blur = st.slider("Blur", 0, 100, 0)
        vignette_strength = st.slider("Vignette", 0, 100, 0)
        film_grain = st.slider("Film Grain", 0, 100, 0)
        effect = st.selectbox("Apply a filter", ["None", "Grayscale", "Pencil Sketch", "Sepia"])

    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.sidebar.expander("✨ Animated Effects"):
        animated_effect = st.selectbox("Add animated weather!", ["None", "Rain", "Snow"])
        animation_frames = st.slider("Animation Frames", 10, 50, 20)
        animation_speed = st.slider("Animation Speed", 0.05, 0.2, 0.1, 0.01)

    if not st.session_state.processing_complete:
        st.markdown('<div class="processing-status">Processing Image...</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        
        processed_img = img_bgr.copy()
        
        # Apply Exposure
        processed_img = cv2.convertScaleAbs(processed_img, alpha=1 + exposure / 100.0, beta=0)
        progress_bar.progress(1/12)
        
        # Apply Highlights & Shadows
        processed_img = adjust_highlights_shadows(processed_img, highlights, shadows)
        progress_bar.progress(2/12)
        
        # Apply Brightness & Contrast
        processed_img = cv2.convertScaleAbs(processed_img, alpha=1 + contrast / 100.0, beta=brightness)
        progress_bar.progress(3/12)
        
        # Apply Gamma
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            processed_img = cv2.LUT(processed_img, table)
        progress_bar.progress(4/12)
        
        # Apply Global Saturation and Hue
        hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(cv2.multiply(s, saturation), 0, 255).astype(np.uint8)
        h = ((h.astype(int) + hue) % 180).astype(np.uint8)
        processed_img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        progress_bar.progress(5/12)
        
        # Apply Temperature and Color Balance
        b, g, r = cv2.split(processed_img)
        r = np.clip(cv2.add(r, red_balance), 0, 255).astype(np.uint8)
        g = np.clip(cv2.add(g, green_balance), 0, 255).astype(np.uint8)
        b = np.clip(cv2.add(b, blue_balance), 0, 255).astype(np.uint8) 
        r = np.clip(cv2.add(r, temp * 0.5), 0, 255).astype(np.uint8) 
        b = np.clip(cv2.subtract(b, temp * 0.5), 0, 255).astype(np.uint8) 
        processed_img = cv2.merge([b, g, r])
        progress_bar.progress(6/12)

        # Apply HSL Adjustments
        processed_img = adjust_hsl(processed_img, hsl_hue, hsl_sat, hsl_light)
        progress_bar.progress(7/12)
        
        # Apply Sharpening
        if sharpness > 0:
            kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 10.0, -1], [-1, -1, -1]])
            processed_img = cv2.filter2D(processed_img, -1, kernel)
        progress_bar.progress(8/12)

        # Apply Blur
        if blur > 0:
            k_size = (blur * 2) + 1
            processed_img = cv2.GaussianBlur(processed_img, (k_size, k_size), 0)
        progress_bar.progress(9/12)

        # Apply Vignette
        if vignette_strength > 0:
            processed_img = apply_vignette(processed_img, vignette_strength)
        progress_bar.progress(10/12)

        # Apply Film Grain
        processed_img = add_film_grain(processed_img, film_grain)
        progress_bar.progress(11/12)

        # Apply Static Filters
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
        progress_bar.progress(12/12)
        
        st.session_state.processed_img_cache = processed_img
        st.session_state.processing_complete = True
        progress_bar.empty()
        
    processed_img = st.session_state.processed_img_cache

    final_output_image = processed_img
    animated_gif_bytes = None
    if animated_effect != "None":
        animated_gif_bytes = generate_animated_gif(final_output_image, animated_effect, 
                                                    num_frames=animation_frames, duration_per_frame=animation_speed)

    if st.session_state.processing_complete:
        st.markdown('<div class="processing-status-complete">Ready</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="image-title">Processed</h3>', unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        if animated_gif_bytes:
            st.image(animated_gif_bytes, use_container_width=True, caption=f'{animated_effect} Effect Applied.')
        else:
            st.image(final_output_image, channels="BGR" if len(final_output_image.shape)==3 else "GRAY", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
        
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if animated_gif_bytes:
        st.sidebar.download_button(label="⬇️ Download Animated GIF", data=animated_gif_bytes, file_name="ImageLab_animation.gif", mime="image/gif", use_container_width=True)
    else:
        st.sidebar.download_button(label="📥 Download Edited Image", data=convert_image_to_bytes(final_output_image), file_name="ImageLab_edited_photo.png", mime="image/png", use_container_width=True)

else:
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="landing-message">Welcome to ImageLab Pro</h2>', unsafe_allow_html=True)
    st.image("https://res.cloudinary.com/demo/image/upload/v1605389650/sample_image.jpg", caption="Elevate your photos with cinematic precision.", width=550)
    st.markdown('<p class="upload-prompt">Upload an image using the sidebar to begin your transformation!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
