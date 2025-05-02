import streamlit as st

st.set_page_config(page_title="Meet the Team", layout="wide")
st.title("👨‍💻 Meet the Team")

team = [
    {
        "name": "Arunendra Singh",
        "photo": "assets/photo.jpeg",
        "enrno": "221B092",
        "email": "singharun10235@gmail.com",
        "bio": "Alice worked on training and optimizing the encoder-decoder model for colorization."
    },
    {
        "name": "Bhawesh Pandey",
        "photo": "assets/photo.jpeg",
        "enrno": "221B092",
        "email": "singharun10235@gmail.com",
        "bio": "Bob designed and built the Streamlit interface for this app."
    },
    {
        "name": "Dheeraj Verma",
        "photo": "assets/photo.jpeg",
        "enrno": "221B092",
        "email": "singharun10235@gmail.com",
        "bio": "Charlie coordinated the project and integrated features like enhancements and post-processing."
    }
]

cols = st.columns(3)
for idx, member in enumerate(team):
    with cols[idx]:
        st.image(member["photo"], use_column_width=True)
        st.subheader(member["name"])
        st.text(member["enrno"])
        st.text(member["email"])
        st.markdown(member["bio"])
