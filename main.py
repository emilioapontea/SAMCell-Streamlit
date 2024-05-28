import time
import streamlit as st
from streamlit_image_comparison import image_comparison
from utils import *
import pandas as pd
import requests
import base64
from io import BytesIO


if 'endpoint_available' not in st.session_state:
    st.session_state.endpoint_available = False
if 'image_comparisons' not in st.session_state:
    st.session_state.image_comparisons = {}
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame(columns=['file name', 'cell count', 'avg cell area', 'confluency', 'avg neighbors'])

@st.cache_data(show_spinner=False)
def df_to_csv():
    return st.session_state['df'].to_csv().encode('utf-8')

@st.cache_data(show_spinner=False)
def pil_to_png(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def init_query():
    API_URL = st.secrets["db_url"]
    API_TOKEN = st.secrets["db_token"]
    headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, headers=headers, json={})
    return response.json()

@st.cache_data(show_spinner=False)
def query(payload):
    API_URL = st.secrets["db_url"]
    API_TOKEN = st.secrets["db_token"]
    headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@st.cache_data(show_spinner=False)
def package_payload(uploaded_files):
    # a list of ndarray
    batch = load_images(uploaded_files)
    # a list of base64 json dictionaries
    images = []
    for image in batch:
        img_dict = {
            'data': base64.b64encode(image.tobytes()).decode('utf-8'),
            'shape': image.shape,
            'dtype': str(image.dtype)
        }
        images.append(img_dict)

    return {"inputs": images}

@st.cache_data(show_spinner=False)
def samcell_query(payload):
    return query(payload)

@st.cache_data(show_spinner=False)
def get_model_outputs(uploaded_files, pred):
    if 'labels' not in pred:
        return [], {}

    labels = pred["labels"]
    # list of ndarray
    outputs = []
    for label in labels:
        # Unpack response (base64 encoding of ndarray)
        encoded_labels = label['data']
        labels_shape = label['shape']
        labels_dtype = label['dtype']
        # Decode Base64 encoded data
        decoded_response = base64.b64decode(encoded_labels)
        # Reconstruct ndarray
        output = np.frombuffer(decoded_response, dtype=np.dtype(labels_dtype))
        output = output.reshape(labels_shape)

        outputs.append(output)

    #! This *MIGHT* mess up the order of names and outputs
    rets = {}
    for uploaded_file, output in zip(uploaded_files, outputs):
        output_rgb = convert_label_to_rainbow(output)
        # cv2.imwrite(f'/content/outputs/proc-orig.png', output_rgb)
        ret = (Image.open(uploaded_file), Image.fromarray(output_rgb))
        rets[uploaded_file.name] = ret

    return outputs, rets

@st.cache_data(show_spinner=False)
def get_batch_segmentation(uploaded_files):
    # a list of ndarray
    batch = load_images(uploaded_files)
    # a list of base64 json dictionaries
    images = []
    for image in batch:
        img_dict = {
            'data': base64.b64encode(image.tobytes()).decode('utf-8'),
            'shape': image.shape,
            'dtype': str(image.dtype)
        }
        images.append(img_dict)

    payload = {"inputs": images}

    pred = query(payload)

    if 'labels' not in pred:
        return [], {}

    labels = pred["labels"]
    # list of ndarray
    outputs = []
    for label in labels:
        # Unpack response (base64 encoding of ndarray)
        encoded_labels = label['data']
        labels_shape = label['shape']
        labels_dtype = label['dtype']
        # Decode Base64 encoded data
        decoded_response = base64.b64decode(encoded_labels)
        # Reconstruct ndarray
        output = np.frombuffer(decoded_response, dtype=np.dtype(labels_dtype))
        output = output.reshape(labels_shape)

        outputs.append(output)

    #! This *MIGHT* mess up the order of names and outputs
    rets = {}
    for uploaded_file, output in zip(uploaded_files, outputs):
        output_rgb = convert_label_to_rainbow(output)
        # cv2.imwrite(f'/content/outputs/proc-orig.png', output_rgb)
        ret = (Image.open(uploaded_file), Image.fromarray(output_rgb))
        rets[uploaded_file.name] = ret

    return outputs, rets

# DEPRECATED
@st.cache_data(show_spinner=False)
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
info = st.empty()
button = st.empty()
if not st.session_state.endpoint_available:
    if button.button(
        'Get Started!',
        help='SAMCell may not currently be running. It may take a few minutes to initialize.',
        type='primary',
        use_container_width=True
        ):
        button.empty()
        with st.spinner('Sit tight! SAMCell is starting up... (this may take a few minutes)'):
            q = None
            while q is None or q['error'] == '503 Service Unavailable':
                info.info("SAMCell is setup to sleep after 15 minutes without requests", icon="🥱")
                q = init_query()
                time.sleep(1)
            info.empty()
            st.session_state.endpoint_available = True
            st.rerun()
else:
    uploaded_files = st.file_uploader(
        label="Select image(s) to segment",
        accept_multiple_files=True,
        type=('.png', '.jpg', '.jpeg', 'tif', 'tiff')
    )

    if uploaded_files:
        if st.button(label=f"Run SAMCell on {len(uploaded_files)} image(s)"):
            # with st.spinner('Sit tight! SAMCell is processing your images...'):
            # For single image request
            # for file in uploaded_files:
            #     st.session_state.image_comparisons[file.name] = get_model_segmentation(file)
            # if not all(st.session_state.image_comparisons.values()):
            #     st.error('Oh oh! SAMCell did not respond to your request! If the issue persists, contact GTPBL to restart the endpoint.', icon="😴")
            #     st.session_state.image_comparisons = {}

            # For batch endpoint requests (preferred)
            # outputs, st.session_state.image_comparisons = get_batch_segmentation(uploaded_files)

            with st.status("Sit tight! SAMCell is processing your images...", expanded=True) as status:
                placeholder = st.empty()
                container = placeholder.container()
                container.write(":package: Packaging payload...")
                payload = package_payload(uploaded_files)
                container.write(":outbox_tray: Sending request...")
                time.sleep(0.2)
                container.write(":thought_balloon: Doing some linear algebra...")
                pred = query(payload)
                container.write(":magic_wand: Making some magic...")
                outputs, st.session_state.image_comparisons = get_model_outputs(uploaded_files, pred)

                if len(st.session_state.image_comparisons.keys()) == 0:
                    status.update(label="SAMCell Error", state="error", expanded=True)
                    placeholder.empty()
                    st.error('Oh oh! SAMCell did not respond to your request! If the issue persists, contact GTPBL to restart the endpoint.', icon="😴")
                    st.session_state.image_comparisons = {}
                else:
                    status.update(label="Segmentation complete!", state="complete", expanded=False)
                    for file, output in zip(uploaded_files, outputs):
                        metrics = compute_metrics(output)

                        # print(f'image: {file.name}')
                        # print(f'\t cell count: {cell_count}')
                        # print(f'\t avg cell area: {cell_area}')
                        # print(f'\t confluency: {confluency}')
                        # print(f'\t avg neighbors: {avg_neighbors}')

                        st.session_state['df'].loc[len(st.session_state['df'])] = (file.name, *metrics)


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

        col1, col2 = st.columns(2)

        with col1:
            download_img = pil_to_png(img2)
            file_prefix = st.text_input(
                "Enter a file download prefix",
                placeholder="Enter a file download prefix",
                label_visibility='collapsed'
            )
        with col2:
            if file_prefix:
                btn = st.download_button(
                    label=f"Download `{file_prefix}-{dropdown}`",
                    data=download_img,
                    file_name=f"{file_prefix}-{dropdown}",
                    mime="image/png"
                )

        if st.button("Show metrics"):
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