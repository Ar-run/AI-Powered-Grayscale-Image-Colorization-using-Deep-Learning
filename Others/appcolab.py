import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import skimage.color as sk
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Image Colorizer",
    page_icon="🎨",
    layout="wide"
)

# App title and description
st.title("🎨 Grayscale to Color Image Colorizer")
st.markdown("""
Upload a grayscale image and watch it transform into color!
This app uses a deep learning model trained on landscape images to colorize grayscale photos.
""")

# Functions for color space conversion
def convert_lab(image):
    lab_image = sk.rgb2lab(image)
    return lab_image

def convert_rgb(image):
    rgb_image = sk.lab2rgb(image)
    return rgb_image

# Load the models
@st.cache_resource
def load_models():
    try:
        encoder_model = tf.keras.models.load_model('Models\encoder_model2.keras')
        decoder_model = tf.keras.models.load_model('Models\decoder_model2.keras')
        return encoder_model, decoder_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

encoder_model, decoder_model = load_models()

# Image preprocessing
def preprocess_image(image, target_size=(224, 224)):
    # Ensure image is RGB
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:  # If RGBA, remove alpha channel
        image = image[:, :, :3]
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert to LAB color space
    lab_image = convert_lab(image)
    
    # Extract L channel
    l_channel = lab_image[:, :, 0]
    
    return image, l_channel

# Colorize function
def colorize_image(l_channel, encoder_model, decoder_model):
    # Reshape L channel for processing
    l_input = l_channel.reshape(1, 224, 224)
    
    # Create a 3-channel grayscale image for VGG16 (which expects 3 channels)
    vgg_input = np.stack((l_channel,) * 3, axis=-1)
    vgg_input = vgg_input.reshape((1, 224, 224, 3))
    
    # Extract features using encoder
    features = encoder_model.predict(vgg_input)
    features = features.reshape(1, 7, 7, 512)
    
    # Predict ab channels
    ab_output = decoder_model.predict(features)
    
    # Scale back to original range
    ab_output = ab_output * 128
    
    # Create and convert the colorized image
    colorized = np.zeros((224, 224, 3))
    colorized[:, :, 0] = l_input[0]
    colorized[:, :, 1:] = ab_output[0]
    
    # Convert from LAB to RGB
    colorized_rgb = convert_rgb(colorized)
    
    return colorized_rgb

# File uploader
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Check if image is grayscale or convert to grayscale
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Convert to grayscale if it's a color image
        grayscale = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        grayscale_3channel = np.stack((grayscale,) * 3, axis=-1)
        original_for_display = grayscale_3channel
    else:
        # Already grayscale
        grayscale = img_array
        grayscale_3channel = np.stack((grayscale,) * 3, axis=-1)
        original_for_display = grayscale_3channel
    
    # Create columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Grayscale Image")
        st.image(original_for_display, use_column_width=True)
    
    if encoder_model is not None and decoder_model is not None:
        # Process and colorize image
        with st.spinner('Colorizing image...'):
            # Preprocess image
            _, l_channel = preprocess_image(grayscale_3channel)
            
            # Colorize
            colorized_image = colorize_image(l_channel, encoder_model, decoder_model)
            
            # Convert to uint8 for display (0-255)
            colorized_image_display = (colorized_image * 255).astype(np.uint8)
            
            # Display colorized image
            with col2:
                st.subheader("Colorized Image")
                st.image(colorized_image_display, use_column_width=True)
            
            # Add download button for colorized image
            colorized_pil = Image.fromarray(colorized_image_display)
            buf = io.BytesIO()
            colorized_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Colorized Image",
                data=byte_im,
                file_name="colorized_image.png",
                mime="image/png"
            )
    else:
        st.error("Models could not be loaded. Please check if they exist in the correct directory.")

# Add information about the model
with st.expander("About the Colorization Model"):
    st.markdown("""
    ### How it works
    
    This colorization model uses a deep learning approach with transfer learning:
    
    1. **Color Space**: We work in the L*a*b* color space, where 'L' represents lightness and 'a*b*' represent the color dimensions
    2. **Feature Extraction**: A pre-trained VGG16 model extracts features from the grayscale image
    3. **Colorization**: A decoder network predicts the color channels ('a*b*') from these features
    4. **Reconstruction**: The predicted color channels are combined with the original lightness channel to create the colorized image
    
    The model was trained on landscape images to learn color patterns and relationships.
    """)

# Add troubleshooting information
with st.expander("Troubleshooting"):
    st.markdown("""
    ### Common Issues
    
    - **Blue-tinted results**: This can happen if the model doesn't generalize well to new images or if preprocessing steps are different
    - **Poor color accuracy**: The model is trained on specific types of images and may not perform well on all image types
    - **Model loading errors**: Ensure the model files are in the same directory as this app
    
    For best results, try using landscape images similar to what the model was trained on.
    """)
