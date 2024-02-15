import streamlit as st

with st.sidebar:
    "Welcome to the sidebar"

st.title("SAMCell")
st.caption("A Cell Segmentation Model powered by Segment Anything Model")

uploaded_file = st.file_uploader("Upload an article", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'))

if uploaded_file:
    st.write("Upload completed!")