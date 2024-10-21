import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *

set_freq = [(20, 20, 1 + 1j), (22, 20, 1 - 2j), (24, 20, 3 + 1j),
                (22, 22, 1 + 3j), (20, 22, 2 - 1j), (20, 24, 2 + 1j),
                (22, 24, 1 + 3j), (24, 22, 1 - 1j), (24, 24, 1 + 2j),
                (25, 25, 1 + 1j), (22, 25, 1 - 2j), (25, 22, 1 + 2j)
                ]


betrag = lambda A: np.sqrt(sum(A**2))
def LinearInterpol(arr, x, y):
    """
    interpoliert zwischen punkten auf einem array
    """
    dx, dy = x - int(x), y - int(y)
    x, y = int(x), int(y)
    r1 = arr[x,y] * (1-dx) + arr[x+1,y] * dx
    r2 = arr[x,y+1] * (1 - dx) + arr[x+1,y+1] * dx
    return r1 * (1 - dy) + r2 * dy

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

    def getCoordinateArray(size=size):
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


        norm_x = corners[2] - corners[0]
        norm_y = corners[1] - corners[0]
        sizex, sizey = abs(betrag(corners[0] - corners[1])), abs(
            betrag(corners[0] - corners[2]))  # assumes square pattern
        size = int(max(sizey, sizex)) if size is None else size

        # Convert the corner points to numpy arrays
        corner1 = corners[0]
        corner2 = corners[1]
        corner3 = corners[2]

        # Calculate the vectors from corner1
        v1 = corner2 - corner1  # Vector from corner1 to corner2
        v2 = corner3 - corner1  # Vector from corner1 to corner3

        t = np.linspace(0, 1, size)

        # Create a grid of all combinations of u and v
        u_grid, v_grid = np.meshgrid(t, t)
        points = np.zeros(shape=(size,size,2))
        for x in range(size):
            for y in range(size):
                #points[y,x] = corner1 + t[x]*v1 + t[y]*v2
                points[y,x] = corner1 + x/size*v1 + y/size*v2
        #print("2:", coordinates[:2,:2])
        #print(np.average(np.abs(coordinates - points)), "average difference in coordinates")
        return coordinates

        # Calculate the points using vector addition and broadcasting
        #points = corners[0] + np.outer(u_grid.ravel(), v1) + np.outer(v_grid.ravel(), v2)
        #points = points.reshape(size, size, 2)
        #print(np.average(np.abs(coordinates - points)))
        #return points

    indizes = getCoordinateArray(size=size)
    if interpolator != scipyBicubic:
        data = np.zeros(shape=(size, size))
        for x in range(size):
            for y in range(size):
                data[x,y] = interpolator(trans, indizes[x,y,1], indizes[x,y,0])
        return data
    else:
        return scipyBicubic(data=trans, idx=indizes)



def correctApproximation(raw, corners_raw):
    approx = invertTransform(raw, corners_raw, interpolator=SignalInterpol)
    approx_fft = irfft2(approx)
