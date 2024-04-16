from typing import Tuple
import numpy as np
from PIL import Image, ImageOps
import torch
# import cv2

def load_images(file_list):
    images = []
    for file in file_list:
        image = load_image(file)
        images.append(image)
    return images

def load_image(file_path):
    image = np.array(ImageOps.grayscale(Image.open(file_path)))
    # image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    max_dimension = 1000
    height, width = image.shape[:2]
    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))

    # resized_image = cv2.resize(image, (new_width, new_height))
    return image

def convert_label_to_rainbow(label: np.ndarray) -> np.ndarray:
    label_rainbow = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cell in np.unique(label):
        if cell == 0:
            continue
        label_rainbow[label == cell] = np.random.rand(3) * 255
    return label_rainbow

# def compute_metrics(output: np.ndarray) -> Tuple[int, float, int, float]:
#     return np.random.randint(0, 100, size=4)

def compute_metrics(output: np.ndarray) -> Tuple[int, float, int, float]:
    #compute cell count
    cell_count = len(np.unique(output)) - 1

    #compute cell area
    total_cell_area = np.sum(output != 0)
    cell_area = total_cell_area / cell_count

    #compute confluency
    confluency = total_cell_area / (output.shape[0] * output.shape[1])

    #compute number of neighbors per cell
    neighbors = []
    for cell in np.unique(output):
        if cell == 0:
            continue
        cell_coords = output == cell
        #add 5 pixel buffer around cell
        cell_coords = np.where(cell_coords)
        cell_coords = (np.clip(cell_coords[0] - 5, 0, output.shape[0] - 1), np.clip(cell_coords[1] - 5, 0, output.shape[1] - 1))

        #get all cells within 10 pixels
        neighbor_cells = np.unique(output[cell_coords[0], cell_coords[1]])
        neighbors.append(len(neighbor_cells) - 1)

    #compute average number of neighbors
    avg_neighbors = np.mean(neighbors)

    #round metrics to 2 decimal places
    cell_count = round(cell_count, 2)
    cell_area = round(cell_area, 2)
    confluency = round(confluency, 2)
    avg_neighbors = round(avg_neighbors, 2)

    #convert confluency to percentage
    confluency *= 100
    # confluency = f'{int(confluency)}%'
    confluency = int(confluency)

    return cell_count, cell_area, confluency, avg_neighbors

class ToyModel(torch.nn.Module):
    def __init__(self):
        return

    def forward(self, data: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(data[:,:,:3]).to(torch.float32)
        inputs = torch.unsqueeze(inputs.permute(2, 0, 1), 0)
        filters = torch.randn(1, 3, 3, 3).to(torch.float32)
        outputs = torch.squeeze(torch.nn.functional.conv2d(inputs, filters, padding=1))
        return outputs.numpy()

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.forward(data)