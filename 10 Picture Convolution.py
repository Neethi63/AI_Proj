import numpy as np
import tensorflow as tf
from skimage import io, color
import matplotlib.pyplot as plt

# Load the group picture and the cropped image using skimage
groupImg = io.imread('group_picture.jpg')
croppedImg = io.imread('AE.jpg')

# Check if the images were loaded successfully
if groupImg is None or croppedImg is None:
    print("Error: Unable to load one of the images.")
    exit()

# Convert both images to grayscale
groupGray = color.rgb2gray(groupImg)
croppedGray = color.rgb2gray(croppedImg)

# Reshape the images to add batch and channel dimensions
groupGrayTensor = tf.convert_to_tensor(groupGray, dtype=tf.float32)
groupGrayTensor = tf.reshape(groupGrayTensor, (1, groupGrayTensor.shape[0], groupGrayTensor.shape[1], 1))

croppedGrayTensor = tf.convert_to_tensor(croppedGray, dtype=tf.float32)
croppedGrayTensor = tf.reshape(croppedGrayTensor, (croppedGrayTensor.shape[0], croppedGrayTensor.shape[1], 1, 1))

# Perform convolution to find the template with stride of 1
resultTensor = tf.nn.conv2d(groupGrayTensor, croppedGrayTensor, strides=[1, 1, 1, 1], padding='VALID')
result = resultTensor.numpy().squeeze()

# Find the location with the maximum match
maxLoc = np.unravel_index(np.argmax(result), result.shape)

# Draw a rectangle around the matched region
topLeft = maxLoc
h, w = croppedGray.shape
bottomRight = (topLeft[0] + h, topLeft[1] + w)

# Display the results using matplotlib
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(groupImg)
rect = plt.Rectangle(topLeft[::-1], w, h, edgecolor='green', facecolor='none', linewidth=2)
ax.add_patch(rect)
plt.title('Matched Image')
plt.show()
