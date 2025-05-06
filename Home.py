import streamlit as st

st.set_page_config(page_title="Image Colorization using Deep Learning", layout="centered")

#for scrolling background image
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-repeat: repeat-x;
        animation: scroll-background 60s linear infinite;
    }

    @keyframes scroll-background {
        from {
            background-position: 0 0;
        }
        to {
            background-position: -1000px 0;
        }
    }

    .title-text {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title-text">', unsafe_allow_html=True)
st.title("🎨 Welcome to Grayscale Image Colorizer")
st.markdown("""
Upload your grayscale photos and watch them come to life using our Deep learning based colorizer!

**Use the sidebar or navigation menu to:**
- Start colorizing your images
- Learn about the colorization process in brief
""")
st.markdown("</div>", unsafe_allow_html=True)
