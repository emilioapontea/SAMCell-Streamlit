import streamlit as st
from streamlit_image_comparison import image_comparison
from utils import *
import requests
import base64

if 'image_comparisons' not in st.session_state:
    st.session_state.image_comparisons = {}

@st.cache_data
def query(payload):
    API_URL = "https://yn8lan37azo8xw3k.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept" : "application/json",
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@st.cache_data
def get_model_segmentation(uploaded_file, new_width=1000):
    # return Image.open(uploaded_file), Image.open('res/proc-orig.png')

    # prepare playoad
    image = load_image(uploaded_file)

    # Convert the ndarray to a bytes object
    data_bytes = image.tobytes()

    # Encode the bytes object using Base64
    encoded_data = base64.b64encode(data_bytes).decode('utf-8')
    payload = {"inputs": {
        'data': encoded_data,
        'shape': image.shape,
        'dtype': str(image.dtype)
    }}

    # query the endpoint
    pred = query(payload)

    if 'labels' not in pred:
        return None

    labels = pred["labels"]
    # Unpack response (base64 encoding of ndarray)
    encoded_labels = labels['data']
    labels_shape = labels['shape']
    labels_dtype = labels['dtype']
    # Decode Base64 encoded data
    decoded_response = base64.b64decode(encoded_labels)
    # Reconstruct ndarray
    output = np.frombuffer(decoded_response, dtype=np.dtype(labels_dtype))
    output = output.reshape(labels_shape)

    output_rgb = convert_label_to_rainbow(output)
    # cv2.imwrite(f'/content/outputs/proc-orig.png', output_rgb)
    return Image.open(uploaded_file), Image.fromarray(output_rgb)

st.title("SAMCell")
st.caption("A Cell Segmentation Model powered by Segment Anything Model  \nDeveloped by the [Georgia Tech Precision Biosystems Lab](https://pbl.gatech.edu/)")

uploaded_files = st.file_uploader(
    label="Select image(s) to segment",
    accept_multiple_files=True,
    type=('.png', '.jpg', '.jpeg', 'tif', 'tiff')
)

if uploaded_files:
    if st.button(label=f"Run SAMCell on {len(uploaded_files)} image(s)"):
        for file in uploaded_files:
            st.session_state.image_comparisons[file.name] = get_model_segmentation(file)
        if not all(st.session_state.image_comparisons.values()):
            st.error('Oh oh! SAMCell did not respond to your request! If the issue persists, contact GTPBL to restart the endpoint.', icon="😴")
            st.session_state.image_comparisons = {}

dropdown = st.selectbox(
    label="Preview segmentation",
    placeholder="Select an image to preview...",
    # index=0,
    options=st.session_state.image_comparisons.keys(),
    # options=[file.name for file in uploaded_files]
)

img1, img = None, None
if dropdown:
    img1, img2 = st.session_state.image_comparisons[dropdown]

# st.write(f'Comparing images: {img1} : {img2}')

if img1 and img2:
    image_comparison(
        img1=img1,
        img2=img2,
        label1=f"Original: {dropdown}",
        label2=f"SAMCell: {dropdown}",
        make_responsive=True,
        in_memory=False
    )

        # st.session_state.dropdown = st.selectbox(
        #     label="Select an image to preview",
        #     index=0,
        #     options=st.session_state.image_comparisons.keys(),
        # )

    # tabs = st.tabs([file.name for file in uploaded_files])
    # image_comparisons = [get_model_segmentation(file) for file in uploaded_files]
    # # for file in uploaded_files:
    # #     img1, img2 = get_model_segmentation(file)
    # #     image_comparisons.append(
    # #         image_comparison(
    # #             img1=img1,
    # #             img2=img2,
    # #             label1=f"Original: {file.name}",
    # #             label2=f"SAMCell: {file.name}",
    # #             make_responsive=True,
    # #             in_memory=False
    # #         )
    # #     )
    # for tab, file, img in zip(tabs, uploaded_files, image_comparisons):
    #     with tab:
    #         img1, img2 = img
    #         image_comparison(
    #             img1=img1,
    #             img2=img2,
    #             label1=f"Original: {file.name}",
    #             label2=f"SAMCell: {file.name}",
    #             make_responsive=True,
    #             in_memory=False
    #         )
    #         # f"{file}"
    #         # f"{file.name}"