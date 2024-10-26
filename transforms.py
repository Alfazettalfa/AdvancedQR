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

#np.random.seed(42872362)
a = 10
size = 400

#runTroughModel(source="FourierDataset\\train_images\\water_body_43.jpg")

#pattern = JapanPattern(size=size)[..., 0]
pattern, original_coeff = getRandomFourier(size=size, RGB=True)
pattern = pattern - np.min(pattern)
pattern = pattern / np.max(pattern) * 255
print(np.mean(pattern, axis=(0, 1)))
print(np.std(pattern, axis=(0,1)))



trans, corners = randomlyProjectPattern(pattern, max_inclination=np.pi/2.5, phi=(0.05, 0.05, 1.2), add_noise=False, sigma=20)

p2 = trans.copy()
r = np.random.rand(*p2.shape) * 255
p2 = np.where(np.less(p2, 1), r, p2)
p2 = np.astype(p2, np.uint8)
p2 = np.dstack((p2[..., np.newaxis], p2[..., np.newaxis], p2[..., np.newaxis]))
plt.imshow(p2)
plt.show()
gaussKernel(p2)

#corners = correctApproximation(raw_image=trans, corners_approx=corners, size=size)
inverted = invertTransform(trans, corners, size=size, interpolator= cubicInterpolTorch)#lambda arr, y, x: SignalInterpol(arr, y, x, methode='hanning'))


inverted_fft = rfft2(inverted, norm='forward')
pattern_fft = rfft2(pattern, norm='forward')
pattern_fft_copy = np.real(np.copy(inverted_fft/np.max(np.abs(inverted_fft))))


differenz = np.abs(np.abs(pattern_fft)) - np.abs(inverted_fft)
differenz = differenz/np.max(np.abs(differenz))
differenz[0,0] = 1
differenz[0,1] = 0


plots = {'inverted fft' : np.real(inverted_fft) / np.max(np.abs(inverted_fft)), 'original fft' : np.real(pattern_fft) / np.max(np.abs(inverted_fft)),
         'trans' : trans, 'original': pattern, 'inverted' : inverted, 'differenz' : differenz,
         }

with open("parameters.json", "r") as f:
    data = json.load(f)
# Convert the JSON data back to a list of tuples with complex numbers
set_freq = [((item['x'], item['y']), complex(item['real'], item['imag'])) for item in data["orienting frequencies"]]
n_frequencies = len(set_freq)
amplitudes = []
frequencies = []
for i in range(len(set_freq)):
    amplitudes.append(set_freq[i][1] / set_freq[0][1])
    frequencies.append(set_freq[i][0])

measured_frequencies = []
for i in range(n_frequencies):
    measured_frequencies.append(getFourierCoefficientTorch(signal=inverted, target_freq_x=frequencies[i][0],
                            target_freq_y=frequencies[i][1]))

zero = measured_frequencies[0]
for i in range(len(measured_frequencies)):
    measured_frequencies[i] = measured_frequencies[i] / zero


print("Baseline: ", amplitudes)
print("Inverted: ", [f"{complex(a):2f}" for a in measured_frequencies])


fig, axes = plt.subplots(1, len(plots.keys()) + 2, figsize=(15, 5))


for ax, img, title in zip(axes, plots.values(), plots.keys()):
    ax.imshow(img, cmap='inferno')
    ax.set_title(title)
    #ax.axis('off')  # Turn off axis
axes[-1].hist(pattern_fft_copy.flatten(), bins=40)
axes[-1].set_title('spectral distribution')
axes[-2].hist(pattern.flatten(), bins=40)
axes[-2].set_title('real distribution')

print(f"ERROR in FFT: {np.average(np.abs(plots['inverted fft'] - plots['original fft'])) / (2/3) * 100:4f}%")
#print(f"ERROR in original: {(np.average(np.abs(plots['inverted']-plots['original']))):4f}")
plt.show()