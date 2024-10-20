import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import sin, cos
import torch
import torchvision
import torchvision.transforms as transforms
import os
from matplotlib.image import imread
import cv2
from skimage import transform
from numpy import sin, cos
from numpy.random import rand
from numpy.fft import *
from scipy.fft import dctn, idctn, dstn, idstn
from transforms import randomTransform
from markerlab import JapanPattern as Jap

randomTransform(pattern=Jap())













spectral = lambda a, b, size=100: (b - 1j*a) * size / 2          #für a*sin + b*cos

size = 1000
hamming_x = np.hamming(size)
hamming_y = np.hamming(size)
hamming_2d = hamming_x[:, np.newaxis] * hamming_y[np.newaxis, :]

x = np.linspace(start = 0, stop = 2 * np.pi, num = size)
y = x[:, np.newaxis]



X = x * np.ones(shape=y.shape)
Y = y * np.ones(shape=x.shape)


f = 10*cos(5*X + 5*Y) + 10*sin(5*X + 20*Y)
f = f# * hamming_2d
a = rfft2(f)

plots = {'real':np.real(a), 'imag':np.imag(a), 'original': f}
fig, axes = plt.subplots(1, len(plots.keys()), figsize=(15, 5))

for ax, img, title in zip(axes, plots.values(), plots.keys()):
    ax.imshow(img, cmap='inferno')
    ax.set_title(title)
    ax.axis('off')  # Turn off axis


plt.show()








