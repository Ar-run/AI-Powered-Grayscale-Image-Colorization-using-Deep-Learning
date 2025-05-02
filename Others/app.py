import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from skimage.color import rgb2lab, lab2rgb
from PIL import Image

# Load saved models
@st.cache_resource
def load_models():
    encoder = tf.keras.models.load_model("encoder_model2.keras")
    decoder = tf.keras.models.load_model("decoder_model2.keras")
    return encoder, decoder

encoder_model, decoder_model = load_models()

# Preprocessing that matches training exactly
def preprocess_for_vgg(image_pil):
    # Convert PIL to OpenCV format
    image = np.array(image_pil.convert("L"))  # grayscale
    image = cv2.resize(image, (224, 224))  # Resize
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    
    # Merge grayscale to pseudo RGB
    pseudo_rgb = cv2.merge((image, image, image))  # (224,224,3)
    pseudo_rgb = pseudo_rgb.reshape(1, 224, 224, 3)

    return image * 100, pseudo_rgb  # Return L channel unnormalized, for LAB merge

def colorize_image(L_orig, pseudo_rgb):
    vgg_feat = encoder_model.predict(pseudo_rgb)
    ab = decoder_model.predict(vgg_feat)[0] * 128

    lab = np.zeros((224, 224, 3))
    lab[:, :, 0] = L_orig
    lab[:, :, 1:] = ab
    rgb = lab2rgb(lab)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb

# Streamlit app interface
st.title("Grayscale to Color - Trained Model Output")
uploaded = st.file_uploader("Upload a grayscale or RGB image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Input Image", use_column_width=True)
    
    L_img, pseudo_rgb = preprocess_for_vgg(pil_img)
    colorized = colorize_image(L_img, pseudo_rgb)

    st.image(colorized, caption="Colorized Output", use_column_width=True)
