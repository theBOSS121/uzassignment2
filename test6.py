# %%
import numpy as np
from UZ_utils import *
from a2_utils import *
import cv2
from matplotlib import pyplot as plt

# %% [markdown]
# # Exercise 3

# %% [markdown]
# ### a

# %%
def simple_convolution(signal, kernel):
    N = int(len(kernel) / 2) # kernel of size 2N + 1
    returnSignal = np.zeros(len(signal))
    for i in range(0, len(signal)): # loop through signal
        for j in range(0, 2*N+1): # loop through kenel
            index = i-(j-N)
            if index < 0: # extending edge
                index = 0
            elif index > len(signal)-1:
                index = len(signal)-1
            returnSignal[i] += kernel[j]*signal[index] # weigted sum

    return returnSignal

# %%
def calculateGaussianKernel(sigma):
    N = int(np.ceil(3 * sigma))
    kernel = np.zeros(2 * N + 1)
    for x in range(-N, N):
        kernel[x+N] = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x**2 / (2 * (sigma**2)))

    return kernel

# %%
def gaussfilter(img):
    gaussianKernal = calculateGaussianKernel(2)
    for i, row in enumerate(img):
        img[i] = simple_convolution(row, gaussianKernal)
    img = img.T
    for i, row in enumerate(img):
        img[i] = simple_convolution(row, gaussianKernal)
    return img.T

temp = cv2.imread('images/lena.png') # 0-255
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
gaussNoise = gauss_noise(temp, 15)
saltPepperNoise = sp_noise(temp)
filteredGaussNoise= gaussfilter(np.copy(gaussNoise))
filteredSaltPepperNoise= gaussfilter(np.copy(saltPepperNoise))

f = plt.figure(figsize=(15, 10))
f.add_subplot(2, 3, 1)
plt.imshow(temp, cmap="gray")
plt.title("Original")
f.add_subplot(2, 3, 2)
plt.imshow(gaussNoise, cmap="gray")
plt.title("Gauss noise")
f.add_subplot(2, 3, 3)
plt.imshow(saltPepperNoise, cmap="gray")
plt.title("Salt and Pepper")
f.add_subplot(2, 3, 5)
plt.imshow(filteredGaussNoise, cmap="gray")
plt.title("Filtered Gauss noise")
f.add_subplot(2, 3, 6)
plt.imshow(filteredSaltPepperNoise, cmap="gray")
plt.title("Filtered Salt and Pepper")
plt.show()


# %% [markdown]
# Question: Which noise is better removed using the Gaussian filter?

# %% [markdown]
# Gauss noise

# %% [markdown]
# ### b

# %%
def sharpenfilter(img):
    # gaussianKernal = calculateGaussianKernel(2)
    a = 0.02
    sharpenKernal = (np.array([0, 1 + a, 0]) - np.array([a/3, a/3, a/3]))
    print(sharpenKernal)
    for i, row in enumerate(img):
        img[i] = simple_convolution(row, sharpenKernal)
    img = img.T
    for i, row in enumerate(img):
        img[i] = simple_convolution(row, sharpenKernal)
    return img.T

temp = cv2.imread('images/museum.jpg') # 0-255
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
sharpendedImage = sharpenfilter(np.copy(temp))

f = plt.figure(figsize=(15, 10))
f.add_subplot(1, 2, 1)
plt.imshow(temp, cmap="gray")
plt.title("Original")
f.add_subplot(1, 2, 2)
plt.imshow(sharpendedImage, cmap="gray")
plt.title("Sharpened")
plt.show()


