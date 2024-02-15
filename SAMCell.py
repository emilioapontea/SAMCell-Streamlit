import os
import streamlit as st
import csv

with st.sidebar:
    "Welcome to the sidebar"

st.title("SAMCell")
st.caption("A Cell Segmentation Model powered by Segment Anything Model")

uploaded_file = st.file_uploader("Upload an article", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'))

if uploaded_file:
    st.sucess(f"Uploaded {uploaded_file.name} successfully.")

# Text input field for user to enter data
# user_input = st.text_input("Enter some text:")
lines_to_write = [
    ['file name', 'cell count', 'avg cell area', 'confluency', 'avg neighbors'],
]

# Button to trigger writing data to file
if st.button("Write to File"):
    # Get the filename from the user
    filename = st.text_input("Enter filename:", "metrics.csv")
    os.makedirs("output", exist_ok=True)

    # Open the file in write mode and write the user input
    with open(os.path.join("output", filename), 'w', newline='') as csvfile:
        # csvfile.write(user_input)
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(lines_to_write)

    # Confirmation message
    st.success(f"Data written to {filename} successfully.")