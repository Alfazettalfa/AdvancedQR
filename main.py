import numpy as np
from matplotlib import pyplot as plt
from skimage import transform
from numpy import sin, cos
from markerlab import *
from numpy.fft import *
import obspy as ob
from reconstructer import *
import json
from models import *
from dataset import generateImageWithKnownborders

np.random.seed(42872362)

size = 400
pattern = getRandomFourierwBorder(size=size)
pattern_image, corners = generateImageWithKnownborders(pattern=pattern)
bitmap = runTroughModel(source=pattern_image.astype(np.float32) / 255)
corners_measured = findParallelogramCornersFromMask(bitmap)
show = np.dstack([bitmap[..., np.newaxis] for _ in range(3)])
pattern_image2 = pattern_image.copy()
vg_corners = []
for c in corners_measured:
    show[c[1]-3:c[1]+3, c[0]-3:c[0]+3, 0] = 255
    pattern_image2[c[1]-10:c[1]+10, c[0]-10:c[0]+10, 0] = 255
    plt.imshow(pattern_image[c[1]-10:c[1]+10, c[0]-10:c[0]+10, :])
    plt.show()
    vg_corners.append(findTargetCorners(pattern_image[c[1]-10:c[1]+10, c[0]-10:c[0]+10, :]))
print(vg_corners)

plots = {'pattern on Image':pattern_image, 'bitmap with corners' : show, 'Image with marked corners' : pattern_image2}


fig, axes = plt.subplots(1, len(plots.keys()), figsize=(15, 5))
for ax, img, title in zip(axes, plots.values(), plots.keys()):
    ax.imshow(img, cmap='inferno')
    ax.set_title(title)

plt.show()