import os
import streamlit as st
import csv
from io_utils import ImageHandler

handler = ImageHandler()

file_names = []

def handle_file_upload(file):
    if file is not None:
        file_names.append(file.name)
        st.success(f"There are now {len(file_names)} files uploaded!")

def main():
    sidebar = st.sidebar
    with sidebar:
        "Welcome to the sidebar"
        filename = st.text_input("Enter filename:", handler._default_csv)

    st.title("SAMCell")
    st.caption("A Cell Segmentation Model powered by Segment Anything Model.  \nDeveloped at the [Georgia Tech Precision Biosystems Lab](https://pbl.gatech.edu/)")

    uploaded_file = st.file_uploader("Upload an article", type=('.png', '.jpg', '.jpeg', 'tif', 'tiff'))

    # st.success(handler.file_upload(uploaded_file))
    handle_file_upload(uploaded_file)


    lines_to_write = [
        ['file name', 'cell count', 'avg cell area', 'confluency', 'avg neighbors'],
    ]

    if st.button("Write to File"):

        # Open the file in write mode and write the user input
        with open(handler.csv_path, 'w', newline='') as csvfile:
            # csvfile.write(user_input)
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(lines_to_write)

        # Confirmation message
        st.success(f"Data written to {filename} successfully.")

if __name__ == "__main__":
    main()