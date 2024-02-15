import streamlit as st
from streamlit_image_comparison import image_comparison
import pandas as pd
import numpy as np
from PIL import Image

@st.cache_data
def df_to_csv(df):
    return df.to_csv().encode('utf-8')

st.title("SAMCell")
st.caption("A Cell Segmentation Model powered by Segment Anything Model")

uploaded_file = st.file_uploader("Upload an article", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'))


if uploaded_file:
    image_comparison(
        img1=Image.open(uploaded_file),
        img2=Image.open(uploaded_file)
    )
    st.success(f"Uploaded {uploaded_file.name} successfully.")

df = pd.DataFrame(columns=['file name', 'cell count', 'avg cell area', 'confluency', 'avg neighbors'])

if st.sidebar.button("Generate random data"):
    df.loc[len(df)] = (np.random.randint(0, 100, size=5))
    st.dataframe(df)

csv = df_to_csv(df)
file_name = "metrics.csv"

if st.download_button(
    label="Download analysis",
    data = csv,
    file_name=file_name,
    mime='text/csv'
):
    # Confirmation message
    st.success(f"Data written to {file_name} successfully.")