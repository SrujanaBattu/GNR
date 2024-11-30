#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FileUpload
import io

# Load the image
img = Image.open('Screenshot 2024-11-28 113244.png')

# Display the image
plt.imshow(img)
plt.show()


# In[2]:


grayscale_image = img.convert("L")  # Convert to grayscale
binary_image = np.array(grayscale_image) > 128  # Thresholding to create binary image

# Display the binary image
plt.figure(figsize=(5, 5))
plt.imshow(binary_image, cmap="gray")
plt.title("Binary Image (Thresholded)")
plt.axis("off")
plt.show()


# In[3]:


from scipy.ndimage import binary_dilation, binary_erosion



# Define a structuring element (3x3 square)
structuring_element = np.ones((3, 3), dtype=bool)

# Manual implementation of morphological operations
def manual_dilation(image, element):
    output = np.zeros_like(image, dtype=bool)
    pad_h, pad_w = element.shape[0] // 2, element.shape[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.any(padded_image[i:i+element.shape[0], j:j+element.shape[1]] & element)
    return output

def manual_erosion(image, element):
    output = np.zeros_like(image, dtype=bool)
    pad_h, pad_w = element.shape[0] // 2, element.shape[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.all(padded_image[i:i+element.shape[0], j:j+element.shape[1]] & element)
    return output

# Perform manual dilation and erosion
manual_dilated = manual_dilation(binary_image, structuring_element)
manual_eroded = manual_erosion(binary_image, structuring_element)

# Compare with built-in functions
scipy_dilated = binary_dilation(binary_image, structure=structuring_element)
scipy_eroded = binary_erosion(binary_image, structure=structuring_element)

# Display results for dilation
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(manual_dilated, cmap='gray')
plt.title("Manual Dilation")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(scipy_dilated, cmap='gray')
plt.title("Built-in Dilation")
plt.axis("off")
plt.show()

# Display results for erosion
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(manual_eroded, cmap='gray')
plt.title("Manual Erosion")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(scipy_eroded, cmap='gray')
plt.title("Built-in Erosion")
plt.axis("off")
plt.show()


# In[5]:


from scipy.ndimage import binary_opening, binary_closing

# Manual implementation of opening (erosion followed by dilation)
def manual_opening(image, element):
    eroded = manual_erosion(image, element)
    opened = manual_dilation(eroded, element)
    return opened

# Manual implementation of closing (dilation followed by erosion)
def manual_closing(image, element):
    dilated = manual_dilation(image, element)
    closed = manual_erosion(dilated, element)
    return closed

# Perform manual opening and closing
manual_opened = manual_opening(binary_image, structuring_element)
manual_closed = manual_closing(binary_image, structuring_element)

# Compare with built-in functions
scipy_opened = binary_opening(binary_image, structure=structuring_element)
scipy_closed = binary_closing(binary_image, structure=structuring_element)

# Display results for opening
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(manual_opened, cmap='gray')
plt.title("Manual Opening")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(scipy_opened, cmap='gray')
plt.title("Built-in Opening")
plt.axis("off")
plt.show()

# Display results for closing
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(manual_closed, cmap='gray')
plt.title("Manual Closing")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(scipy_closed, cmap='gray')
plt.title("Built-in Closing")
plt.axis("off")
plt.show()


# In[6]:


# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(manual_dilated, cmap='gray')
plt.title("Manual Dilation")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(scipy_dilated, cmap='gray')
plt.title("Built-in Dilation")
plt.axis("off")

plt.show()


# In[7]:


import time

start_manual = time.time()
manual_dilation(binary_image, structuring_element)
time_manual = time.time() - start_manual

start_scipy = time.time()
binary_dilation(binary_image, structure=structuring_element)
time_scipy = time.time() - start_scipy

print(f"Manual Method Time: {time_manual:.5f}s")
print(f"Built-in Method Time: {time_scipy:.5f}s")


# In[ ]:




