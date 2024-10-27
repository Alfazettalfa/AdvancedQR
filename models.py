from skimage import transform
from numpy import sin, cos
from reconstructer import *
from modelHelpers.config import ALL_CLASSES
from modelHelpers.model import prepare_model
from modelHelpers.utils import get_segment_labels, draw_segmentation_map, image_overlay
import numpy as np

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

def runTroughModel(source, model_name='FourierWithCornerDetector.pth', return_array=True):
    from PIL import Image
    import numpy as np
    if isinstance(source, str):
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
    outputs = outputs[..., 0]
    return outputs.astype(np.uint8)

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



def randomlyProjectPattern(pattern, max_inclination=np.pi / 5, phi=None, add_noise=False, sigma=3, RGB=False):
    if phi is None:
        α, β, γ = (np.random.rand(3) * 2 - 1) * np.array([max_inclination, max_inclination, np.pi])
    else:
        α, β, γ = phi
    rotation = np.array([
        [cos(β) * cos(γ), sin(α) * sin(β) * cos(γ) - cos(α) * sin(γ), 0],
        [cos(β) * sin(γ), sin(α) * sin(β) * sin(γ) + cos(α) * cos(γ), 0],
        [0, 0, 1]
    ])

    if RGB:
        back = np.zeros(shape=(pattern.shape[0] * 3, pattern.shape[1] * 3, 3))
        back[pattern.shape[0] * 1:pattern.shape[0] * 2, pattern.shape[1] * 1:pattern.shape[1] * 2, :] = pattern
    else:
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
    return back.astype(np.uint8), border + (np.random.normal(scale=sigma, loc=0, size=border.shape) if add_noise else 0)

def findParallelogramCornersFromMask(mask):
    """
    Find the corners of a parallelogram in a binary mask.

    Parameters:
    - mask: A binary mask (2D NumPy array) where the parallelogram region is marked as 1 or 255,
            and the background is 0.

    Returns:
    - A NumPy array of four points (x, y) representing the corners in consistent order:
      [top-left, top-right, bottom-right, bottom-left].
    """
    import cv2
    # Ensure the mask is binary (0 and 255)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming this is the parallelogram)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour as a polygon with a precision factor
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Ensure we have four corners (for a parallelogram)
    if len(approx) != 4:
        raise ValueError("The detected shape does not have four corners. Adjust parameters or verify the input image.")

    # Extract coordinates of the corners
    corners = np.array([point[0] for point in approx])

    # Sort corners: top-left, top-right, bottom-right, bottom-left
    # Calculate the centroid of the points to help sorting
    centroid = np.mean(corners, axis=0)

    # Sort based on position relative to the centroid
    sorted_corners = sorted(corners, key=lambda p: (np.arctan2(p[1] - centroid[1], p[0] - centroid[0])))

    # Rearrange points to a consistent order
    # Top-left, top-right, bottom-right, bottom-left
    top_left, top_right, bottom_right, bottom_left = sorted_corners

    return np.array([top_left, top_right, bottom_right, bottom_left])


def findTargetCorners(img):
    def check_violet_to_other_color_transition(patch):
        """
        Checks if there's a transition from violet to another color (background or gray) in the given patch.
        """
        # Define example RGB ranges for colors (adjust based on actual colors in the image)
        violet_range = ((120, 0, 120), (255, 80, 255))  # Approximate violet range
        gray_range = ((100, 100, 100), (150, 150, 150))  # Approximate gray range

        # Check if the patch contains both violet and another target color
        violet_pixels = ((patch[:, :, 0] >= violet_range[0][0]) & (patch[:, :, 0] <= violet_range[1][0]) &
                         (patch[:, :, 1] >= violet_range[0][1]) & (patch[:, :, 1] <= violet_range[1][1]) &
                         (patch[:, :, 2] >= violet_range[0][2]) & (patch[:, :, 2] <= violet_range[1][2]))

        gray_pixels = ((patch[:, :, 0] >= gray_range[0][0]) & (patch[:, :, 0] <= gray_range[1][0]) &
                       (patch[:, :, 1] >= gray_range[0][1]) & (patch[:, :, 1] <= gray_range[1][1]) &
                       (patch[:, :, 2] >= gray_range[0][2]) & (patch[:, :, 2] <= gray_range[1][2]))

        # Detect if there is a mix of violet and gray
        if np.any(violet_pixels) and np.any(gray_pixels):
            return True
        return False
    import cv2
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    # Convert the image to RGB if it is BGR (as OpenCV loads in BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img

    # Convert to grayscale for edge and corner detection
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Edge detection using Canny to highlight borders
    edges = cv2.Canny(gray, 50, 150)

    # Corner detection using Harris corner detection
    corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    # Threshold for Harris corners
    img_rgb[corners > 0.01 * corners.max()] = [0, 255, 0]  # Mark detected corners in green for visualization

    # Detecting precise corners with sub-pixel accuracy
    ret, thresh = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)
    thresh = np.uint8(thresh)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # Refine corner locations to sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    precise_corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Filtering corners based on color transitions
    target_corners = []
    print(precise_corners)
    for corner in precise_corners:
        x, y = int(corner[0]), int(corner[1])

        # Extract color values around the corner (small 3x3 region)
        patch = img_rgb[max(0, y - 1):y + 2, max(0, x - 1):x + 2]

        # Calculate average colors to identify regions
        avg_color = np.mean(patch.reshape(-1, 3), axis=0)

        # Define color thresholds (adjust as needed for your image)
        violet_threshold = [120, 0, 120]  # Example threshold for violet (RGB)
        gray_threshold = [128, 128, 128]  # Example threshold for gray (RGB)

        # Check if the patch meets violet-to-background or violet-to-gray transition
        if np.all(avg_color >= violet_threshold) and np.all(avg_color <= [255, 80, 255]):
            # For violet-to-background or violet-to-gray transition
            if check_violet_to_other_color_transition(patch):
                target_corners.append(corner)

    # Visualization
    print(target_corners)
    plt.imshow(img_rgb)
    plt.scatter([c[0] for c in precise_corners], [c[1] for c in precise_corners], color='yellow', s=40, marker='x')
    plt.title("Detected Target Corners")
    plt.show()

    return target_corners


