import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageEnhance
import skimage.color as sk
import io

st.set_page_config(page_title="Colorizer v2", layout="wide")
st.title("🎨 Grayscale Image Colorizer")

# Set background color
st.markdown(
    """
    Upload a grayscale image and watch it transform into color!
    This app uses a deep learning model trained on landscape images to colorize grayscale photos.
    """
    
    """
    <style>
    .main {
        background-color: #f2e0c9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Detect image upload change and reset sliders
def reset_adjustments():
    st.session_state.brightness = 1.0
    st.session_state.contrast = 1.0
    st.session_state.sharpness = 1.0
    st.session_state.temperature = 0
    st.session_state.color_intensity = 1.0

# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], on_change=reset_adjustments)

# Sidebar sliders with default values
st.sidebar.header("Image Adjustments")
color_intensity = st.sidebar.slider("Color Intensity", 0.0, 2.0, 1.0, 0.1, key="color_intensity")
brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1, key="brightness")
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1, key="contrast")
sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0, 0.1, key="sharpness")
temperature = st.sidebar.slider("Color Temperature", -50, 50, 0, 1, key="temperature")


# Load models
@st.cache_resource
def load_models():
    try:
        encoder_model = tf.keras.models.load_model('Models/encoder_model2.keras')
        decoder_model = tf.keras.models.load_model('Models/decoder_model2.keras')
        return encoder_model, decoder_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

encoder_model, decoder_model = load_models()

def convert_lab(image):
    return sk.rgb2lab(image)

def convert_rgb(image):
    return sk.lab2rgb(image)

def preprocess_image(image, target_size=(224, 224)):
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    image = cv2.resize(image, target_size)
    lab_image = convert_lab(image)
    l_channel = lab_image[:, :, 0]
    return image, l_channel

def colorize_image(l_channel, encoder_model, decoder_model, intensity=1.0):
    l_input = l_channel.reshape(1, 224, 224)
    vgg_input = np.stack((l_channel,) * 3, axis=-1).reshape((1, 224, 224, 3))
    features = encoder_model.predict(vgg_input).reshape(1, 7, 7, 512)
    ab_output = decoder_model.predict(features) * 128 * intensity
    ab_output = np.clip(ab_output, -128, 127)
    colorized = np.zeros((224, 224, 3))
    colorized[:, :, 0] = l_input[0]
    colorized[:, :, 1:] = ab_output[0]
    return convert_rgb(colorized)

def adjust_temperature(img, shift):
    img = img.astype(np.int16)
    img[:, :, 0] = np.clip(img[:, :, 0] + shift, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] - shift, 0, 255)
    return img.astype(np.uint8)

# Main app logic
if encoder_model and decoder_model:
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            grayscale = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            grayscale = img_array

        grayscale_rgb = np.stack((grayscale,) * 3, axis=-1)
        _, l_channel = preprocess_image(grayscale_rgb)
        colorized_image = colorize_image(l_channel, encoder_model, decoder_model, intensity=color_intensity)
        colorized_image = (colorized_image * 255).astype(np.uint8)

        # Enhancements
        enhanced = Image.fromarray(colorized_image)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
        enhanced = np.array(enhanced)
        enhanced = adjust_temperature(enhanced, temperature)

        # Side-by-side comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Grayscale")
            st.image(grayscale_rgb, use_column_width=True)
        with col2:
            st.subheader("Colorized + Enhanced")
            st.image(enhanced, use_column_width=True)

        # Download button
        buf = io.BytesIO()
        Image.fromarray(enhanced).save(buf, format='PNG')
        st.download_button("Download Colorized Image", data=buf.getvalue(), file_name="colorized_image.png", mime="image/png")
    else:
        st.info("Upload a grayscale image to begin.")
else:
    st.error("Models could not be loaded.")
