import numpy as np
from matplotlib import pyplot as plt
from skimage import transform
from numpy import sin, cos
from markerlab import *
from numpy.fft import *
import obspy as ob
from reconstructer import *
import json

np.random.seed(42872362)
def randomlyProjectPattern(pattern, max_inclination = np.pi/5, phi = None, add_noise = False, sigma=None):
    if phi is None:
        α, β, γ = (np.random.rand(3) * 2 - 1) * np.array([max_inclination, max_inclination, np.pi])
    else:
        α, β, γ = phi
    rotation = np.array([
        [cos(β) * cos(γ), sin(α) * sin(β) * cos(γ) - cos(α) * sin(γ), 0],
        [cos(β) * sin(γ), sin(α) * sin(β) * sin(γ) + cos(α) * cos(γ), 0],
        [0, 0, 1]
    ])

    back = np.zeros(shape=(pattern.shape[0]*3, pattern.shape[1]*3))
    back[pattern.shape[0]*1:pattern.shape[0]*2, pattern.shape[1]*1:pattern.shape[1]*2] = pattern
    #back = np.astype(back, np.uint8)
    border = np.array([[pattern.shape[0]*1,pattern.shape[1]*1, 1], [pattern.shape[0]*2, pattern.shape[1]*1, 1],
                       [pattern.shape[0]*1, pattern.shape[1]*2, 1]]).T

    rotation = transform.EuclideanTransform(matrix=rotation)
    shift = transform.EuclideanTransform(translation=-np.array(back.shape[:2]) / 2)
    scale = transform.SimilarityTransform(scale= np.random.rand() + 1)
    matrix = np.linalg.inv(shift.params) @ rotation.params @ scale.params @ shift.params
    tform = transform.EuclideanTransform(matrix)
    back = transform.warp(back, tform.inverse, preserve_range=True, order=1)
    #back = np.astype(back, np.uint8)
    border = tform.params @ border
    border = border[:-1, ...].T
    return back, border + (np.random.normal(scale=sigma, loc=0, size=border.shape) if add_noise else 0)


size = 100


#pattern = JapanPattern(size=size)[..., 0]
pattern, original_coeff = getRandomFourier(size=size)
trans, corners = randomlyProjectPattern(pattern, max_inclination=np.pi/2.5, phi=(0.05, 0.05, 1.2), add_noise=False, sigma=1)
inverted = invertTransform(trans, corners, size=size, interpolator= cubicInterpolTorch)#lambda arr, y, x: SignalInterpol(arr, y, x, methode='hanning'))

print('original average: ', np.average(pattern))
print('inverted average: ', np.average(inverted))
inverted_fft = rfft2(inverted, norm='forward')
pattern_fft = rfft2(pattern, norm='forward')

pattern_fft_copy = np.real(np.copy(inverted_fft/np.max(np.abs(inverted_fft))))
pattern_fft = np.abs(pattern_fft)
inverted_fft = np.abs(inverted_fft)
inverted_fft = inverted_fft / np.max(np.abs(inverted_fft))
pattern_fft = pattern_fft / np.max(np.abs(pattern_fft))

differenz = np.abs(pattern_fft - inverted_fft)
differenz[0,0] = 1
differenz[0,1] = 0


plots = {'inverted fft' : inverted_fft, 'original fft' : pattern_fft, 'trans' : trans,
         'original': pattern, 'inverted' : inverted, 'differenz' : differenz,
         }

with open("parameters.json", "r") as f:
    data = json.load(f)
# Convert the JSON data back to a list of tuples with complex numbers
set_freq = [((item['x'], item['y']), complex(item['real'], item['imag'])) for item in data["orienting frequencies"]]
print(set_freq)
n_frequencies = len(set_freq)
amplitudes = []
frequencies = []
for i in range(len(set_freq)):
    amplitudes.append(set_freq[i][1] / set_freq[0][1])
    frequencies.append(set_freq[i][0])

measured_frequencies = []
for i in range(n_frequencies):
    print(frequencies[i][0], frequencies[i][1])
    measured_frequencies.append(getFourierCoefficient(signal=inverted, target_freq_x=frequencies[i][0],
                            target_freq_y=frequencies[i][1]))
print("Baseline: ", amplitudes)
print("Inverted: ", measured_frequencies)


fig, axes = plt.subplots(1, len(plots.keys()) + 2, figsize=(15, 5))


for ax, img, title in zip(axes, plots.values(), plots.keys()):
    print(title)
    ax.imshow(img) #, cmap='inferno')
    ax.set_title(title)
    #ax.axis('off')  # Turn off axis
axes[-1].hist(pattern_fft_copy.flatten(), bins=40)
axes[-1].set_title('spectral distribution')
axes[-2].hist(pattern.flatten(), bins=40)
axes[-2].set_title('real distribution')

print(f"ERROR in FFT: {np.average(np.abs(plots['inverted fft'] - plots['original fft'])) / (2/3) * 100:4f}%")
#print(f"ERROR in original: {(np.average(np.abs(plots['inverted']-plots['original']))):4f}")
plt.show()