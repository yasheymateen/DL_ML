from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('data/im')
print(type(im))

arr = np.array(im)
print(arr)

arr.shape
plt.imshow(arr)
plt.imshow(im)
gray = arr.mean(axis=2)
gray.shape
plt.imshow(gray)
plot.imshow(gray, cmap='gray')

