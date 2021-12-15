import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import random
from scipy import ndimage
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import datasets, layers, models, losses
from keras.preprocessing.image import ImageDataGenerator

# Define function to read 3D images
def read_file(path):
    # Read file using nibabel
    scan = nib.load(path)
    # Get data
    scan = scan.get_fdata()
    return scan

# Define function to normalize pixel intensity (Hounsfield units)
# Relavant pixels range from -1024 to 400 HU values, which are normalized to be between 0 and 1
def normalize(image):
    # Define range
    min = -1024
    max = 400
    # Assign pixels with intensity outside range to min or max values
    image[image < min] = min
    image[image > max] = max
    # Normalize
    image = (image - min) / (max - min)
    image = image.astype("float32")
    return image

def resize_image(image):
    # Set the desired size of image
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current size of image
    current_depth = image.shape[-1]
    current_width = image.shape[0]
    current_height = image.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    image = ndimage.rotate(image, 90, reshape=False)
    # Resize across z-axis
    image = ndimage.zoom(image, (width_factor, height_factor, depth_factor), order=1)
    return image

def process_scan(path):
    # Read scan
    image = read_file(path)
    # Normalize
    image = normalize(image)
    # Resize image using previously defined resize_image(image)
    image = resize_image(image)
    return image

# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
normal_scan_paths = [
    os.path.join(os.getcwd(), ".\COVID19_1110\studies\CT-0", x)
    for x in os.listdir(".\COVID19_1110\studies\CT-0")
]

# Folder "CT-1 and CT-2" consist of CT scans havinginvolvement of lung parenchyma is less than 25% and ground-glass opacifications,
# and involvement of lung parenchyma is between 25 and 50% respectively
abnormal_scan_paths = [
    os.path.join(os.getcwd(), ".\COVID19_1110\studies\CT-1", x)
    for x in os.listdir(".\COVID19_1110\studies\CT-1")] + [os.path.join(os.getcwd(), ".\COVID19_1110\studies\CT-2", x)
    for x in os.listdir(".\COVID19_1110\studies\CT-2")
]

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

# Randomly scramble data
abnormal_scans = shuffle(abnormal_scans)
normal_scans = shuffle(normal_scans)

print(normal_scans.shape)
print(abnormal_scans.shape)

# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

print(len(abnormal_scans))
print(len(normal_scans))

Percent_abn = len(abnormal_scans)*70//100
Percent_n = len(normal_scans)*70//100

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((abnormal_scans[:Percent_abn], normal_scans[:Percent_n]), axis=0)
y_train = np.concatenate((abnormal_labels[:Percent_abn], normal_labels[:Percent_n]), axis=0)
x_val = np.concatenate((abnormal_scans[Percent_abn:], normal_scans[Percent_n:]), axis=0)
y_val = np.concatenate((abnormal_labels[Percent_abn:], normal_labels[Percent_n:]), axis=0)

# Randomly scramble data and labels
x_train, y_train = shuffle(x_train, y_train)
x_val, y_val = shuffle(x_val, y_val)

print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

print(len(x_train))

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

batch_size = 4

datagen = ImageDataGenerator(
        rotation_range = 90,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True)

training_generator = datagen.flow(x_train, y_train, batch_size = batch_size)

print(len(training_generator))

steps_per_epoch = len(x_train)//batch_size

model = models.Sequential([
  layers.Conv2D(4, 3, padding = 'same', activation = 'relu', input_shape = (128, 128, 64)),
  layers.MaxPooling2D(),
  layers.Dropout(0.8),
  layers.Conv2D(8, 3, padding = 'same', activation = 'relu', input_shape = (128, 128, 64)),
  layers.MaxPooling2D(),
  layers.Dropout(0.8),
  layers.Flatten(),
  layers.Dense(8, activation = 'relu'),
  layers.Dense(2, activation = 'softmax')
])

epochs = 200
model.summary()

model.compile(optimizer = 'adam',                                                     # Gradient descent method
              loss = 'sparse_categorical_crossentropy',                               # Computes the crossentropy loss between the labels and predictions
              metrics = ['accuracy'])

# Train model
history = model.fit(training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

np.save('my_history.npy', history.history)

model.save(".\model.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Looss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('Loss_acc.png')
plt.show()