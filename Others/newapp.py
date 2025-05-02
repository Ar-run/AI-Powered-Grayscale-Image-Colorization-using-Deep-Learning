import numpy as np
import cv2
import streamlit as st
from cv2 import dnn
from PIL import Image

# --------Model file paths--------#
proto_file = 'colorization_deploy_v2.prototxt'
model_file = 'colorization_release_v2.caffemodel'
hull_pts = 'pts_in_hull.npy'
# ------------------------------#

# Load model parameters and kernel
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Function to perform colorization
def colorize_image(uploaded_file):
    # Read and preprocess the image
    img = np.array(Image.open(uploaded_file))
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize the image for the network
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Predict the ab channels from the L channel
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted 'ab' volume to the same dimensions as our input image
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    # Take the L channel from the image
    L = cv2.split(lab_img)[0]

    # Join the L channel with the predicted ab channel
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

    # Convert the image from Lab to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Change the image to 0-255 range and convert it from float32 to uint8
    colorized = (255 * colorized).astype("uint8")

    # Resize the images for displaying
    img = cv2.resize(img, (640, 640))
    colorized = cv2.resize(colorized, (640, 640))

    # Concatenate images side by side
    result = cv2.hconcat([img, colorized])

    return result

# Streamlit app interface
st.title("Image Colorization with Deep Learning")
st.write("Upload a grayscale image to colorize it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Colorize the uploaded image
    colorized_img = colorize_image(uploaded_file)

    # Display the result
    st.image(colorized_img, channels="BGR", caption="Colorized Image")
