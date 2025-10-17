import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import io
import time
import imageio

st.set_page_config(
    page_title="ImageLab Pro",
    page_icon="🎬", # Cinematic icon
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

# --- NEW: Cinematic Attribute Functions ---
def adjust_highlights_shadows(img, highlights=0, shadows=0):
    if highlights == 0 and shadows == 0:
        return img
    
    # Convert to LAB color space for better H&S control
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    # Adjust shadows
    if shadows != 0:
        shadow_mask = (L < 128).astype(np.float32) * (shadows / 100.0)
        L = np.clip(L + L * shadow_mask, 0, 255).astype(np.uint8)

    # Adjust highlights
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
    
    # Blend grain with image
    grained_img = cv2.addWeighted(img, 1.0, grain, 0.2, 0)
    return grained_img

def adjust_hsl(img, h_shift=0, s_mult=1.0, l_mult=1.0):
    if h_shift == 0 and s_mult == 1.0 and l_mult == 1.0:
        return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Hue shift
    if h_shift != 0:
        h = ((h.astype(int) + h_shift) % 180).astype(np.uint8)
    
    # Saturation
    s = np.clip(s * s_mult, 0, 255).astype(np.uint8)
    
    # Lightness (Value in HSV is often used for lightness/brightness)
    v = np.clip(v * l_mult, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


# --- Animation Generation Functions (from previous version) ---
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

# --- Streamlit App Layout ---
st.markdown('<div class="header"><h1>🎬 ImageLab Pro</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Crafted by <strong>Chetan</strong>. Your professional studio for advanced digital image processing.</p>', unsafe_allow_html=True)

st.sidebar.header("🛠️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload your image to begin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Custom loading spinner
    st.markdown("""
        <div class="loading-container">
            <div class="spinner"></div>
            <p class="loading-text">Loading image...</p>
        </div>
        <style>
            .loading-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 200px;
            }
            .spinner {
                border: 8px solid #f3f3f3;
                border-top: 8px solid #3b82f6;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }
            .loading-text {
                font-family: 'Montserrat', sans-serif;
                font-size: 1.2em;
                color: #4a5568;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        """, unsafe_allow_html=True)
    time.sleep(1) # Simulate loading time
    st.empty() # Clear the spinner
    
    original_image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(original_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.markdown('<div class="main-container">', unsafe_allow_html=True) # Full page fade-in effect
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<h3 class="image-title">Original</h3>', unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True, channels="RGB")
        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True) # Animated divider
    
    # --- NEW: Cinematic Controls ---
    with st.sidebar.expander("🎥 Cinematic Controls", expanded=True):
        st.markdown("<strong>Exposure & Tone</strong>", unsafe_allow_html=True)
        exposure = st.slider("Exposure", -50, 50, 0, key='exposure_slider')
        highlights = st.slider("Highlights", -50, 50, 0, key='highlights_slider')
        shadows = st.slider("Shadows", -50, 50, 0, key='shadows_slider')
        
        st.markdown("<strong>Color Grading (HSL)</strong>", unsafe_allow_html=True)
        hsl_hue = st.slider("HSL Hue Shift", -180, 180, 0, key='hsl_hue_slider')
        hsl_sat = st.slider("HSL Saturation", 0.0, 3.0, 1.0, 0.1, key='hsl_sat_slider')
        hsl_light = st.slider("HSL Lightness", 0.0, 2.0, 1.0, 0.1, key='hsl_light_slider')

    with st.sidebar.expander("⚪ Basic Adjustments"):
        brightness = st.slider("Brightness", -100, 100, 0, key='brightness_slider')
        contrast = st.slider("Contrast", -100, 100, 0, key='contrast_slider')
        gamma = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1, key='gamma_slider')

    with st.sidebar.expander("🎨 Global Color & Tone"):
        saturation = st.slider("Saturation (Global)", 0.0, 3.0, 1.0, 0.1, key='global_saturation_slider')
        hue = st.slider("Hue Shift (Global)", -180, 180, 0, key='global_hue_slider')
        temp = st.slider("Temperature", -100, 100, 0, key='temp_slider')
        st.markdown("<strong>Color Balance</strong>", unsafe_allow_html=True)
        red_balance = st.slider("Red", -50, 50, 0, key='red_balance_slider')
        green_balance = st.slider("Green", -50, 50, 0, key='green_balance_slider')
        blue_balance = st.slider("Blue", -50, 50, 0, key='blue_balance_slider')
        
    with st.sidebar.expander("🌟 Static Effects & Filters"):
        sharpness = st.slider("Sharpness", 0, 100, 0, key='sharpness_slider')
        blur = st.slider("Blur", 0, 100, 0, key='blur_slider')
        vignette_strength = st.slider("Vignette", 0, 100, 0, key='vignette_slider')
        film_grain = st.slider("Film Grain", 0, 100, 0, key='film_grain_slider') # New Film Grain
        effect = st.selectbox("Apply a filter", ["None", "Grayscale", "Pencil Sketch", "Sepia"], key='effect_selector')

    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True) # Animated divider
    with st.sidebar.expander("✨ Animated Effects"):
        animated_effect = st.selectbox(
            "Add animated weather to your photo!",
            ["None", "Rain", "Snow"],
            key='animated_effect_selector'
        )
        animation_frames = st.slider("Animation Frames", 10, 50, 20, key='anim_frames_slider')
        animation_speed = st.slider("Animation Speed", 0.05, 0.2, 0.1, 0.01, key='anim_speed_slider')


    # --- Image Processing Pipeline ---
    st.markdown('<div class="processing-status">Processing Image...</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    processed_img = img_bgr.copy()
    current_step_list = [0] 
    # Update total steps for progress bar
    processing_steps = 15 # More steps due to new cinematic adjustments

    def update_progress_fix():
        current_step_list[0] += 1
        progress_bar.progress(current_step_list[0] / processing_steps)
        time.sleep(0.05)

    # Apply Exposure
    processed_img = cv2.convertScaleAbs(processed_img, alpha=1 + exposure / 100.0, beta=0)
    update_progress_fix()

    # Apply Highlights & Shadows
    processed_img = adjust_highlights_shadows(processed_img, highlights, shadows)
    update_progress_fix()

    # Apply Brightness & Contrast
    processed_img = cv2.convertScaleAbs(processed_img, alpha=1 + contrast / 100.0, beta=brightness)
    update_progress_fix()

    # Apply Gamma
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed_img = cv2.LUT(processed_img, table)
    update_progress_fix()

    # Apply Global Saturation and Hue
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(cv2.multiply(s, saturation), 0, 255).astype(np.uint8)
    h = ((h.astype(int) + hue) % 180).astype(np.uint8)
    processed_img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    update_progress_fix()
    
    # Apply Temperature and Color Balance
    b, g, r = cv2.split(processed_img)
    r = np.clip(cv2.add(r, red_balance), 0, 255).astype(np.uint8)
    g = np.clip(cv2.add(g, green_balance), 0, 255).astype(np.uint8)
    b = np.clip(cv2.add(b, blue_balance), 0, 255).astype(np.uint8) 
    r = np.clip(cv2.add(r, temp * 0.5), 0, 255).astype(np.uint8) 
    b = np.clip(cv2.subtract(b, temp * 0.5), 0, 255).astype(np.uint8) 
    processed_img = cv2.merge([b, g, r])
    update_progress_fix()

    # Apply HSL Adjustments (after global color, but before static effects)
    processed_img = adjust_hsl(processed_img, hsl_hue, hsl_sat, hsl_light)
    update_progress_fix()

    # Apply Sharpening
    if sharpness > 0:
        kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness / 10.0, -1], [-1, -1, -1]])
        processed_img = cv2.filter2D(processed_img, -1, kernel)
    update_progress_fix()

    # Apply Blur
    if blur > 0:
        k_size = (blur * 2) + 1
        processed_img = cv2.GaussianBlur(processed_img, (k_size, k_size), 0)
    update_progress_fix()

    # Apply Vignette
    if vignette_strength > 0:
        processed_img = apply_vignette(processed_img, vignette_strength)
    update_progress_fix()

    # Apply Film Grain
    processed_img = add_film_grain(processed_img, film_grain)
    update_progress_fix()

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
    update_progress_fix()
    
    final_output_image = processed_img

    animated_gif_bytes = None
    if animated_effect != "None":
        st.markdown(f'<div class="processing-status">Generating {animated_effect} Animation...</div>', unsafe_allow_html=True)
        animated_gif_bytes = generate_animated_gif(final_output_image, animated_effect, 
                                                    num_frames=animation_frames, duration_per_frame=animation_speed)
        update_progress_fix() # One more step for animation generation


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
    
    st.markdown('</div>', unsafe_allow_html=True) # Close main-container
        
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True) # Animated divider
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
    st.markdown('<h2 class="landing-message">Welcome to ImageLab Pro</h2>', unsafe_allow_html=True)
    st.image("https://res.cloudinary.com/demo/image/upload/v1605389650/sample_image.jpg", caption="Elevate your photos with cinematic precision.", width=550)
    st.markdown('<p class="upload-prompt">Upload an image using the sidebar to begin your transformation!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
