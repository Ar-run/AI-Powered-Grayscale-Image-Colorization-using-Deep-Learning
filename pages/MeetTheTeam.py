import streamlit as st

st.set_page_config(page_title="Meet the Team", layout="wide")
st.title("👨‍💻 Meet the Team")

st.markdown(
    
    
    """
    <style>
    .main {
        background-color: #f2e0c9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

team = [
    {
        "name": "Arunendra Singh",
        "photo": "assets/arunendra.jpg",
        "enrno": "221B092",
        "email": "vaibhav10235@gmail.com"
    },
    {
        "name": "Bhawesh Pandey",
        "photo": "assets/bhawesh.jpg",
        "enrno": "221B126",
        "email": "bhaweshpandey841@gmail.com"
    },
    {
        "name": "Dheeraj Verma",
        "photo": "assets/dheeraj.jpg",
        "enrno": "221B145",
        "email": "dheerajverma.cp@gmail.com"
    }
]

cols = st.columns(3)
for idx, member in enumerate(team):
    with cols[idx]:
        st.image(member["photo"], use_column_width=True)
        st.subheader(member["name"])
        st.text(member["enrno"])
        st.text(member["email"])
