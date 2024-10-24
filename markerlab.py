import numpy as np
from matplotlib import pyplot as plt
import json

with open("parameters.json", "r") as f:
    data = json.load(f)
# Convert the JSON data back to a list of tuples with complex numbers
set_freq = [((item['x'], item['y']), complex(item['real'], item['imag'])) for item in data["orienting frequencies"]]

def JapanPattern(size=250):
    pattern = np.zeros(shape=(size, size, 3), dtype=np.uint8)
    for i in range(size):
        x = i - size // 2
        #print(int(100*i/size[0]), '%')
        for j in range(size):
            y = j - size // 2
            a = np.arctan2(y, x)
            if a < 0:
                a += 2*np.pi
            r = np.sqrt((x**2 + y**2)/(size**2 + size**2))
            pattern[i,j][0] = int(255/2 + np.sin(a*10) * 255/2)
            pattern[i,j][1] = int(255/2 + np.sin(r*100) * 255/2)
            f = 0 if max([np.sin(x/size * 50), np.sin(y/size * 50)]) < .7 else 1
            pattern[i,j][2] = int(255 * f)
            #if abs(x) + abs(y) < size[0]//13:
             #   pattern[i,j] = 255
              #  if abs(x) < size[0]//75 or abs(y) < size[1]//75:
               #     pattern[i,j] = 0
    return pattern

def getRandomFourier(size=250, set_freq_amplification=0.5):
    """
    :param set_freq_amplification: amplification factor of the preset frequencies
    :param siz e: size of the output array
    :return: The inverse fourier transform of an array of size = size and uniformly distributed fourier coefficients
    """

    from numpy.fft import irfft2, ifft2, rfft2, fft2

    coefficients = (np.random.rand(size, size//2 + 1)*2-1) + 1j * (np.random.rand(size, size//2 + 1)*2-1)
    coefficients = coefficients / np.max(np.abs(coefficients))

    for (u, v), val in set_freq:
        coefficients[u, v] = val / (1 + 3 + 1)**0.5 * set_freq_amplification

    arr = irfft2(coefficients)#, norm='forward')
    arr = arr / np.max(np.abs(arr))
    l = []
    #for i in range(len(set_freq)):
    #    print(coefficients[set_freq[i][0]] / coefficients[set_freq[0][0]])

   # plt.imshow(arr)#, cmap='inferno')
    #plt.show()

    return arr, coefficients




 #plt.imshow(getRandomFourier(), cmap='inferno')
#plt.show()












