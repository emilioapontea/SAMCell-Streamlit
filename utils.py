from typing import Tuple
import numpy as np
from PIL import Image

def convert_label_to_rainbow(label: np.array) -> np.array:
    label_rainbow = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cell in np.unique(label):
        if cell == 0:
            continue
        label_rainbow[label == cell] = np.random.rand(3) * 255
    return label_rainbow

def computeMetrics(output: np.array) -> Tuple[int, float, str, float]:
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
    confluency = f'{int(confluency)}%'

    return cell_count, cell_area, confluency, avg_neighbors

# class ToyModel(torch.nn.Module):
#     def __init__(self):
#         return

#     def eval(self, data: np.array) -> np.array:
#         inputs = torch.from_numpy(data)
#         # sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#         # depth = t.size()[1]
#         # channels = t.size()[2]
#         # sobel_kernel = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(depth, 1, channels, 3, 3)
#         filters = torch.randn(8, 4, 3, 3)
#         inputs = torch.randn(1, 4, 5, 5)
#         outputs = torch.nn.functional.conv2d(inputs, filters, padding=1)
#         return outputs.numpy()