import streamlit as st
from streamlit_image_comparison import image_comparison
import pandas as pd
import numpy as np
from PIL import Image
from utils import *

import requests
from base64 import b64encode

@st.cache_data
def df_to_csv():
    return st.session_state['df'].to_csv().encode('utf-8')

@st.cache_data
def get_model_segmentation(uploaded_file, new_width):
    return Image.open(uploaded_file), Image.open('res/proc-orig.png')

# @st.cache_data
# def get_model_segmentation(uploaded_file, new_width):
#     api_key = "SG_302ea47036b5e98f" #! DO NOT COMMIT
#     url = "https://api.segmind.com/v1/sam-img2img"

#     # Request payload
#     data = {
#     "image": b64encode(uploaded_file.read()).decode('utf-8')
#     }

#     print(f"Requesting SAM API on {uploaded_file.name}...")
#     response = requests.post(url, json=data, headers={'x-api-key': api_key})
#     if response.status_code == 200:
#         # If the response contains JSON data, you can decode it
#         json_response = response.json()
#         # Now you can access specific fields in the JSON response
#         # For example:
#         print(json_response)
#     else:
#         # If the request was not successful, print the status code and reason
#         print("Request failed with status code:", response.status_code)
#         print("Reason:", response.reason)
#         model = ToyModel()
#         input_image = Image.open(uploaded_file)
#         output_image = model(np.array(input_image))
#         width_percent = (new_width / float(input_image.size[0]))
#         new_height = int((float(input_image.size[1]) * float(width_percent)))

#     return input_image.resize((new_width, new_height)), output_image

@st.cache_data
def append_metrics(file, _output):
    # metrics = compute_metrics(output)
    metrics = [95, 1452.62, 20, 3.18]
    st.session_state['df'].loc[len(st.session_state['df'])] = (file.name, *metrics)
    st.session_state['imgs'][file.name] = file
    return metrics

def percentage_circle(label, percentage, size=100):
    percentage = min(max(percentage, 0), 100)
    circumference = 2 * 3.141592 * (size / 2 - 10)
    progress = circumference * (percentage / 100)
    color = st.get_option("theme.primaryColor")
    color = "#ff4b4b" if not color else color

        # <circle cx="{size/2}" cy="{size/2}" r="{size/2 - 10}" stroke="#d3d3d3" stroke-width="3" fill="none" />
    return f"""
    {label}  \n
    <svg width="{size}" height="{size}">
        <circle cx="{size/2}" cy="{size/2}" r="{size/2 - 10}" stroke={color} stroke-width="5" fill="none"
            stroke-dasharray="{progress} {circumference}" stroke-dashoffset="0" transform="rotate(-90 {size/2} {size/2})"
            stroke-linecap="round"/>
        <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Source Sans Pro" font-size="{size * 0.3}" fill={color}>{percentage}%</text>
    </svg>
    """

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame(columns=['file name', 'cell count', 'avg cell area', 'confluency', 'avg neighbors'])
if 'imgs' not in st.session_state:
    st.session_state['imgs'] = {}

if st.sidebar.button("New session"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame(columns=['file name', 'cell count', 'avg cell area', 'confluency', 'avg neighbors'])
    if 'imgs' not in st.session_state:
        st.session_state['imgs'] = {}

st.title("SAMCell")
st.caption("A Cell Segmentation Model powered by Segment Anything Model  \nDeveloped by the [Georgia Tech Precision Biosystems Lab](https://pbl.gatech.edu/)")

# uploaded_file = st.file_uploader("Upload an image", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'))
# get_model_segmentation(uploaded_file)

# TODO: Multiple files
uploaded_files = st.file_uploader(
    "Upload an image",
    type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'),
    accept_multiple_files=True,
)
if uploaded_files:
    tabs = st.tabs([file.name for file in uploaded_files])
    for tab, file in zip(tabs, uploaded_files):
        with tab:
            img1, img2 = get_model_segmentation(file, 1000)
            cell_count, cell_area, confluency, avg_neighbors = append_metrics(file, img2)
            image_comparison(
                img1=img1,
                img2=img2,
                label1=f"Original: {file.name}",
                label2=f"SAMCell: {file.name}",
                make_responsive=True,
                in_memory=False
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cell Count", cell_count)
            col2.metric("Cell Area", cell_area)
            col3.markdown(percentage_circle("Confluency", confluency, size=80), unsafe_allow_html=True)
            col4.metric("Average Neighbors", avg_neighbors)


if st.sidebar.button("Show metrics"):
    st.dataframe(
        st.session_state['df'],
        column_config={
            "file name": "File",
            "cell count": "Cell Count",
            "avg cell area": "Average Cell Area",
            "confluency": st.column_config.ProgressColumn(
                "Confluency",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
            "avg neighbors": "Average Neighbors"
        },
        hide_index=True
    )

csv = df_to_csv()
file_name = "metrics.csv"

if st.download_button(
    label="Download analysis",
    data = csv,
    file_name=file_name,
    mime='text/csv'
):
    # Confirmation message
    st.success(f"Data written to {file_name} successfully.")