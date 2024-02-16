import streamlit as st
from streamlit_image_comparison import image_comparison
import pandas as pd
import numpy as np
from PIL import Image
from utils import *

@st.cache_data
def df_to_csv(df):
    return df.to_csv().encode('utf-8')

@st.cache_data
def get_model_segmentation(uploaded_file):
    if uploaded_file:
        model = ToyModel()
        input_image = Image.open(uploaded_file)
        output_image = model(np.array(input_image))
        image_comparison(
            img1=input_image,
            img2=output_image
        )

st.title("SAMCell")
st.caption("A Cell Segmentation Model powered by Segment Anything Model")

uploaded_file = st.file_uploader("Upload an image", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'))
get_model_segmentation(uploaded_file)

# TODO: Multiple files
# uploaded_files = st.file_uploader("Upload an image", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'), accept_multiple_files=True)
# if uploaded_files:
#     for file in uploaded_files:
#         model = ToyModel()
#         data =

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