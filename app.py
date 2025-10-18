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

def generate_rain_frame(img,
