from typing import List, Tuple
import numpy as np
import cv2
import os


class ImageHandler:
    _instance = None

    def __init__(self, out_directory: str = 'output'):
        os.makedirs(out_directory, exist_ok=True)
        self._output_dir = out_directory
        self._default_csv = "metrics.csv"
        self._uploaded_imgs = []
        self.len = 0

    def __new__(cls):
        if cls._instance is None:
            print("CREATED NEW IMAGEHANDLER")
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def csv_path(self):
        return os.path.join(self._output_dir, self._default_csv)

    # @property
    # def uploaded_imgs(self):
    #     return self._uploaded_imgs

    def file_upload(self, uploaded_file) -> str:
        self.len += 1
        if uploaded_file:
            # sidebar.tabs([img.name for img in handler.append(uploaded_file)])
            self._uploaded_imgs.append(uploaded_file)
            self.print_img_names()
            return f"Uploaded {uploaded_file.name} successfully."
        return f"Something went wrong..."

    def print_img_names(self):
        # ret = "["
        # for img in self._uploaded_imgs:
        #     ret += img.name + ","
        # ret += "]"
        # print(ret)
        print(self.len)

    def load_images(self, directory: str, valid_extensions: List[str] = ['.png']) -> Tuple[list, List[str]]:
        return