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
def get_model_segmentation(uploaded_file, new_width):
    print("PERFORMING COMPARISON")
    model = ToyModel()
    input_image = Image.open(uploaded_file)
    output_image = model(np.array(input_image))
    width_percent = (new_width / float(input_image.size[0]))
    new_height = int((float(input_image.size[1]) * float(width_percent)))
    return input_image.resize((new_width, new_height)), output_image

st.title("SAMCell")
st.caption("A Cell Segmentation Model powered by Segment Anything Model  \nDeveloped by the [Georgia Tech Precision Biosystems Lab](https://pbl.gatech.edu/)")

# uploaded_file = st.file_uploader("Upload an image", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'))
# get_model_segmentation(uploaded_file)

# TODO: Multiple files
uploaded_files = st.file_uploader("Upload an image", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'), accept_multiple_files=True)
if uploaded_files:
    tabs = st.tabs([file.name for file in uploaded_files])
    for tab, file in zip(tabs, uploaded_files):
        with tab:
            img1, img2 = get_model_segmentation(file, 1000)
            image_comparison(
                img1=img1,
                img2=img2,
                label1=f"Original: {file.name}",
                label2=f"SAMCell: {file.name}",
                make_responsive=True,
                in_memory=False
            )

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