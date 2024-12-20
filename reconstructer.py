import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *
import torch


betrag = lambda A: np.sqrt(sum(A**2))


def draw_parallelogram(image, corners):
    from PIL import Image, ImageDraw
    #print(corners)
    image = image + np.min(image)
    image = image / np.max(image) * 100
    image = np.astype(image, np.uint8)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("L")
    draw = ImageDraw.Draw(image)
    corners_array = np.array(corners)
    fourth_corner = corners_array[2] + corners_array[1] - corners_array[0]
    parallelogram_points = [tuple(corners_array[1]), tuple(corners_array[0]), tuple(corners_array[2]),
                            tuple(fourth_corner)]
    draw.polygon(parallelogram_points, outline="black", fill=None, width=1)
    return image
def getCoordinateArray(corners, size):
    norm_x = corners[2] - corners[0]
    norm_y = corners[1] - corners[0]
    sizex, sizey = abs(betrag(corners[0] - corners[1])), abs(
        betrag(corners[0] - corners[2]))  # assumes square pattern
    size = int(max(sizey, sizex)) if size is None else size
    coordinates = np.zeros(shape=(size, size, 2))
    for i in range(size):
        x = norm_x * i / size
        for j in range(size):
            y = norm_y * j / size
            p = x + y + corners[0]
            coordinates[i, j, 0] = p[0]
            coordinates[i, j, 1] = p[1]
    return coordinates

def getCoordinateArrayTorch(corners, size, device, return_numpy=False):
    # Convert corners to PyTorch tensors
    if not torch.is_tensor(corners):
        corners = torch.tensor(corners, dtype=torch.double, requires_grad=True).to(device)

    # Calculate the vectors along the edges
    norm_x = corners[2] - corners[0]  # Vector from corner 0 to corner 2
    norm_y = corners[1] - corners[0]  # Vector from corner 0 to corner 1

    coordinates = torch.zeros((size, size, 2), dtype=torch.double, requires_grad=False).to(device)
    for i in range(size):
        x = norm_x * i / size  # Calculate scaled x-component
        for j in range(size):
            y = norm_y * j / size  # Calculate scaled y-component
            p = x + y + corners[0]  # Final point p for this grid position
            coordinates[i, j, 0] = p[0]  # Assign x-coordinate
            coordinates[i, j, 1] = p[1]  # Assign y-coordinate

    return coordinates.numpy() if return_numpy else coordinates

def getCoordinateArrayTorch_Vectorized(corners, size, device, return_numpy=False):
    # Convert corners to PyTorch tensors
    if not torch.is_tensor(corners):
        corners = torch.tensor(corners, dtype=torch.double, requires_grad=True).to(device)

    # Calculate the vectors along the edges
    norm_x = corners[2] - corners[0]  # Vector from corner 0 to corner 2
    norm_y = corners[1] - corners[0]  # Vector from corner 0 to corner 1

    # Create a grid of coordinates scaled from 0 to 1
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, size, dtype=torch.double, device=device),
                                    torch.linspace(0, 1, size, dtype=torch.double, device=device), indexing='ij')

    # Compute the coordinates using vectorized operations
    coordinates = (grid_x.unsqueeze(-1) * norm_x + grid_y.unsqueeze(-1) * norm_y + corners[0]).to(device)

    return coordinates.cpu().numpy() if return_numpy else coordinates
def SignalInterpol(arr, y, x, a = 3, methode = 'hanning'):
    """
    interpoliert zwischen punkten auf einem array. Erhält die Frequenzen
    besser als LinearInterpol
    """
    from numpy import sinc

    lanczos_kernel = lambda z, off: sinc(z) * np.sinc(z / off)
    hanning_kernel = lambda z, off: sinc(z) * (1 + np.cos(np.pi * z / a)) / 2
    blackman_kernel = lambda z, off: sinc(z) * (21/50 + 0.5 * np.cos(np.pi * z / off) + 12.5 * np.cos(2 * np.pi * z / off))

    kernel_collection = {'blackman' : blackman_kernel, 'lanczos' : lanczos_kernel, 'hanning' : hanning_kernel}
    kernel = kernel_collection[methode]

    x_int, y_int = int(x), int(y)
    value = 0
    norm_factor = 0
    for m in range(y_int - a, y_int + a + 1):
        for n in range(x_int - a, x_int + a + 1):
            weight = kernel(y - m, a) * kernel(x - n, a)
            value += arr[m, n] * weight
            norm_factor += weight
    return value / norm_factor

def cubicInterpol(data, y, x):
    def cubic_kernel(t):
        t = abs(t)
        if t <= 1:
            return (1.5 * t ** 3) - (2.5 * t ** 2) + 1
        elif t <= 2:
            return (-0.5 * t ** 3) + (2.5 * t ** 2) - (4 * t) + 2
        else:
            return 0

    x_int = int(np.floor(x))
    y_int = int(np.floor(y))

    dx = x - x_int
    dy = y - y_int

    # Initialize the interpolated value
    interpolated_value = 0.0

    for m in range(-1, 3):  # Iterate over y dimension
        for n in range(-1, 3):  # Iterate over x dimension
            # Find the integer coordinates of the neighbor point
            x_neighbor = x_int + n
            y_neighbor = y_int + m

            # Ensure the neighbor is within the bounds of the array
            neighbor_value = data[y_neighbor, x_neighbor]

            weight_x = cubic_kernel(dx - n)
            weight_y = cubic_kernel(dy - m)
            interpolated_value += neighbor_value * weight_x * weight_y

    return interpolated_value

def cubicInterpolTorch(data, y, x):
    """
    Perform cubic interpolation for 2D data using 2D floating-point arrays of coordinates (x, y).
    Args:
    - data (torch.Tensor): Input 2D data array.
    - y (torch.Tensor): 2D array of y coordinates (floating-point values).
    - x (torch.Tensor): 2D array of x coordinates (floating-point values).

    Returns:
    - torch.Tensor: Interpolated values at the given (x, y) coordinates.
    """
    def cubic_kernel(t):
        t = torch.abs(t)
        return torch.where(
            t <= 1,
            (1.5 * t ** 3) - (2.5 * t ** 2) + 1,
            torch.where(
                t <= 2,
                (-0.5 * t ** 3) + (2.5 * t ** 2) - (4 * t) + 2,
                torch.zeros_like(t)
            )
        )
    # Ensure x and y are tensors with requires_grad if needed
    if not torch.is_tensor(x):
        x = torch.tensor(x, requires_grad=True, dtype=torch.double)
    if not torch.is_tensor(y):
        y = torch.tensor(y, requires_grad=True, dtype=torch.double)

    # Get the integer parts of the coordinates
    x_int = torch.floor(x).long()  # Integer part of x
    y_int = torch.floor(y).long()  # Integer part of y

    # Get the fractional parts (for interpolation weights)
    dx = x - x_int.double()
    dy = y - y_int.double()

    # Precompute the 16 neighboring indices for all points
    neighbors_x = torch.stack([x_int + i for i in range(-1, 3)], dim=-1).clamp(0, data.shape[1] - 1)
    neighbors_y = torch.stack([y_int + i for i in range(-1, 3)], dim=-1).clamp(0, data.shape[0] - 1)

    # Gather the 16 neighboring values for all points
    neighbor_values = torch.stack(
        [data[neighbors_y[..., i], neighbors_x[..., j]] for i in range(4) for j in range(4)],
        dim=-1
    ).reshape(*x.shape, 4, 4)

    # Compute weights for x and y using the cubic kernel
    weights_x = torch.stack([cubic_kernel(dx - i) for i in range(-1, 3)], dim=-1)  # (*batch_size, 4)
    weights_y = torch.stack([cubic_kernel(dy - i) for i in range(-1, 3)], dim=-1)  # (*batch_size, 4)
    weights_y = weights_y.double()
    weights_x = weights_x.double()

    # Apply weights along the x and y dimensions using matrix multiplication
    # First, apply the weights along the x-axis (columns), then along the y-axis (rows)
    neighbor_values = neighbor_values.to(torch.double)
    weighted_x = torch.matmul(neighbor_values, weights_x.unsqueeze(-1))  # (*batch_size, 4, 1)
    interpolated_values = torch.matmul(weights_y.unsqueeze(-2), weighted_x).squeeze(-1).squeeze(-1)

    return interpolated_values

def scipyBicubic(idx, data):
    from scipy.interpolate import RegularGridInterpolator
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    interpolator = RegularGridInterpolator((y, x), data, method='cubic')
    plt.imshow(data)
    plt.show()
    interpolated_values = interpolator(np.array([[150.1,150.1]]))
    return interpolated_values

def invertTransform(trans, corners, interpolator=cubicInterpol, size=None):
    """
    :param size:
    :param interpolator:
    :param trans: transformed pattern
    :param corners: 3 corners of the transformed
    :return: pattern with inverted transforamtion
    """

    indizes = getCoordinateArray(corners, size=size)
    if interpolator == cubicInterpolTorch:
        indizes = torch.from_numpy(indizes)
        return cubicInterpolTorch(data=torch.from_numpy(trans), x=indizes[...,0], y=indizes[...,1])
    if interpolator != scipyBicubic:
        data = np.zeros(shape=(size, size))
        for x in range(size):
            for y in range(size):
                data[x,y] = interpolator(trans, indizes[x,y,1], indizes[x,y,0])
        return data
    else:
        return scipyBicubic(data=trans, idx=indizes)

def getFourierCoefficientTorch(signal, target_freq_x, target_freq_y, device='cpu'):
    """
    Calculate the Fourier coefficient for a specific target frequency in 2D from the input signal.
    Assumes equidistant sampling in both dimensions.

    Used to get alignement with known frequencies

    Args:
    - signal (torch.Tensor): The input 2D signal (2D tensor).
    - target_freq_x (float): The target frequency index along the x-axis (in cycles per length).
    - target_freq_y (float): The target frequency index along the y-axis (in cycles per length).

    Returns:
    - Fourier coefficient (torch.complex): The Fourier coefficient at the target 2D frequency.
    """
    # Dimensions of the 2D signal
    M, N = signal.shape

    # Generate the indices for x and y directions (equidistant grid)
    x = torch.linspace(0, M - 1, M, dtype=torch.double, device=signal.device, requires_grad=True).to(device)
    y = torch.linspace(0, N - 1, N, dtype=torch.double, device=signal.device, requires_grad=True).to(device)

    # Create a meshgrid for x and y
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Angular frequencies for the target frequencies
    omega_x = 2 * torch.pi * target_freq_x / M  # Normalize frequency by grid size in x
    omega_y = 2 * torch.pi * target_freq_y / N  # Normalize frequency by grid size in y

    # Calculate the real and imaginary parts using the DFT formula
    real_part = torch.sum(signal * torch.cos(omega_x * X + omega_y * Y)) * (4.0 / (M * N))
    imag_part = -torch.sum(signal * torch.sin(omega_x * X + omega_y * Y)) * (4.0 / (M * N))

    # Return the complex Fourier coefficient
    return real_part + 1j * imag_part

def getFourierCoefficient(signal, target_freq_x, target_freq_y):
    if torch.is_tensor(signal):
        signal = signal.numpy()
    M, N = signal.shape

    # Generate the indices for x and y directions (equidistant grid)
    x = np.linspace(0, M - 1, M)
    y = np.linspace(0, N - 1, N)

    # Create a meshgrid for x and y
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Angular frequencies for the target frequencies
    omega_x = 2 * np.pi * target_freq_x / M  # Normalize frequency by grid size in x
    omega_y = 2 * np.pi * target_freq_y / N  # Normalize frequency by grid size in y
    # Calculate the real and imaginary parts using the DFT formula
    real_part = np.sum(signal * np.cos(omega_x * X + omega_y * Y)) * (4.0 / (M * N))
    imag_part = -np.sum(signal * np.sin(omega_x * X + omega_y * Y)) * (4.0 / (M * N))

    # Return the complex Fourier coefficient
    return real_part + 1j * imag_part

def correctApproximation(raw_image, corners_approx, steps=300, size=100):
    """
    :param raw_image:
    :param corners_approx:
    :param steps:
    :return:
    """
    import torch.fft
    import json
    import torch.optim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print("device: ", device)
    with open("parameters.json", "r") as f:
        data = json.load(f)
    # Convert the JSON data back to a list of tuples with complex numbers
    set_freq = [((item['x'], item['y']), complex(item['real'], item['imag'])) for item in data["orienting frequencies"]]
    n_frequencies = len(set_freq)
    amplitudes = []
    frequencies = []
    zero_amp = set_freq[0][1]
    for i in range(len(set_freq)):
        amplitudes.append(set_freq[i][1])# / zero_amp)
        frequencies.append(set_freq[i][0])
    amplitudes = torch.tensor(amplitudes, dtype=torch.complex128, requires_grad=False).to(device)
    frequencies = torch.tensor(frequencies, dtype=torch.int, requires_grad=False).to(device)
    raw_image = torch.tensor(raw_image, dtype=torch.double, requires_grad=False).to(device)

    corners = torch.tensor(corners_approx, dtype=torch.double, requires_grad=True).to(device)
    corners = torch.nn.Parameter(corners)
    #optimizer = torch.optim.SGD([corners], lr=0.1)
    optimizer = torch.optim.SGD([corners], lr=50)

    loss_fn = torch.nn.MSELoss()

    for iter in range(steps):
        plt.imshow(draw_parallelogram(image=raw_image.detach().cpu().numpy(), corners=corners.detach().cpu().numpy()),
                   cmap='nipy_spectral')
        plt.show(block=False)
        plt.pause(.1)
        plt.cla()
        idx_array = getCoordinateArrayTorch_Vectorized(corners, size=size, device=device)
        approx_data = cubicInterpolTorch(raw_image, x=idx_array[...,0], y=idx_array[...,1])
        measured_amplitudes = []
        for i in range(n_frequencies):
            measured_amplitudes.append(getFourierCoefficientTorch(signal=approx_data, target_freq_x=frequencies[i][0],
                                      target_freq_y=frequencies[i][1], device=device))
        measured_amplitudes = torch.stack(measured_amplitudes)  # Stacking to form a tensor
        #zero_amp = measured_amplitudes[0]
        #measured_amplitudes = measured_amplitudes / zero_amp

        #print(measured_amplitudes.detach().cpu().numpy())
        optimizer.zero_grad()
        #loss = loss_fn(torch.real(measured_amplitudes), torch.real(amplitudes))
        #loss = loss + loss_fn(torch.imag(measured_amplitudes), torch.imag(amplitudes))
        loss = loss_fn(torch.angle(measured_amplitudes), torch.angle(amplitudes))
        loss.backward()
        print("Iteration: ", iter ,"   loss: ", loss.detach().cpu().numpy())
        print("corners: ",  corners.detach().cpu().numpy().flatten())
        print("grad: ", corners.grad.detach().cpu().numpy().flatten())
        optimizer.step()
    corners = corners.detach().cpu().numpy()
    return corners