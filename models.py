from skimage import transform
from numpy import sin, cos
from reconstructer import *
from modelHelpers.config import ALL_CLASSES
from modelHelpers.model import prepare_model
from modelHelpers.utils import get_segment_labels, draw_segmentation_map, image_overlay

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

def runTroughModel(source, model_name='FourierWithCornerDetector.pth', return_array=True):
    if isinstance(source, str):
        from PIL import Image
        import numpy as np
        source = Image.open(source)
        source = np.array(source)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model(len(ALL_CLASSES))
    ckpt = torch.load('models/' + model_name)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    # Resize very large images (if width > 1024.) to avoid OOM on GPUs.
    # if image.size[0] > 1024:
    #   image = image.resize((800, 800))
    outputs = get_segment_labels(source, model, device)
    # Get the data from the `out` key.
    outputs = outputs['out']
    outputs = draw_segmentation_map(outputs)
    #outputs = image_overlay(outputs, source)
    return outputs

def gaussKernel(src, mean=128, std=27, kernel_size=(20,20, 3)):
    # Function to calculate z-scores in each kernel window for each channel
    def calculate_z_scores(window, mean, std):
        z_scores = np.abs((window - mean) / std)
        return z_scores.sum(axis=(0, 1))  # Sum across height and width for each channel

    response_map = np.zeros((src.shape[0] - kernel_size[0] + 1, src.shape[1] - kernel_size[1] + 1))
    for i in range(response_map.shape[0]):
        print(i)
        for j in range(response_map.shape[1]):
            window = src[i:i + kernel_size[0], j:j + kernel_size[1], :]

            # Calculate z-scores for the window and sum them
            z_scores = calculate_z_scores(window, mean, std)

            # Aggregate the z-scores for R, G, B channels (e.g., mean or sum)
            response_map[i, j] = np.mean(z_scores)
    plt.imshow(response_map)
    plt.show()



def randomlyProjectPattern(pattern, max_inclination=np.pi / 5, phi=None, add_noise=False, sigma=3):
    if phi is None:
        α, β, γ = (np.random.rand(3) * 2 - 1) * np.array([max_inclination, max_inclination, np.pi])
    else:
        α, β, γ = phi
    rotation = np.array([
        [cos(β) * cos(γ), sin(α) * sin(β) * cos(γ) - cos(α) * sin(γ), 0],
        [cos(β) * sin(γ), sin(α) * sin(β) * sin(γ) + cos(α) * cos(γ), 0],
        [0, 0, 1]
    ])

    back = np.zeros(shape=(pattern.shape[0] * 3, pattern.shape[1] * 3))
    back[pattern.shape[0] * 1:pattern.shape[0] * 2, pattern.shape[1] * 1:pattern.shape[1] * 2] = pattern
    # back = np.astype(back, np.uint8)
    border = np.array([[pattern.shape[0] * 1, pattern.shape[1] * 1, 1], [pattern.shape[0] * 2, pattern.shape[1] * 1, 1],
                       [pattern.shape[0] * 1, pattern.shape[1] * 2, 1]]).T

    rotation = transform.EuclideanTransform(matrix=rotation)
    shift = transform.EuclideanTransform(translation=-np.array(back.shape[:2]) / 2)
    scale = transform.SimilarityTransform(scale=np.random.rand() + 1)
    matrix = np.linalg.inv(shift.params) @ rotation.params @ scale.params @ shift.params
    tform = transform.EuclideanTransform(matrix)
    back = transform.warp(back, tform.inverse, preserve_range=True, order=1)
    # back = np.astype(back, np.uint8)
    border = tform.params @ border
    border = border[:-1, ...].T
    return back, border + (np.random.normal(scale=sigma, loc=0, size=border.shape) if add_noise else 0)

