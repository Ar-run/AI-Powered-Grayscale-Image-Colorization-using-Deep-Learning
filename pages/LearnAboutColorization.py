import streamlit as st

st.set_page_config(page_title="Learn About Colorization", layout="centered")
st.title("🧠 Learn About the Colorization Model")

with st.expander("About the Colorization Model", expanded=True):
    st.markdown("""
    ### How it works

    This colorization model uses a deep learning approach with transfer learning:

    1. **Color Space**: The image is converted to the L*a*b* color space, where:
        - `L` represents lightness (grayscale information),
        - `a` and `b` represent the color components.

    2. **Feature Extraction**: A pre-trained **VGG16** convolutional neural network is used to extract high-level features from the grayscale image.

    3. **Colorization**: A custom decoder network takes these features and predicts the missing `a` and `b` channels (color information).

    4. **Reconstruction**: The predicted `a*b*` values are combined with the original `L` channel, and the result is converted back to RGB color space to generate the final image.

    ---
    
    The model was trained on landscape images, enabling it to learn the natural color distributions in scenery like skies, trees, water, and more.
    """)
