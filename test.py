import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib
import random
from scipy import ndimage
from tensorflow import keras
from tensorflow.python.keras import datasets, layers, models, losses

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    print('1')
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
normal_scan_paths = [
    os.path.join(os.getcwd(), ".\COVID19_1110\studies\CT-0", x)
    for x in os.listdir(".\COVID19_1110\studies\CT-0")
]

# Folder "CT-1 to CT-4" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
abnormal_scan_paths = [
    os.path.join(os.getcwd(), ".\COVID19_1110\studies\CT-1", x)
    for x in os.listdir(".\COVID19_1110\studies\CT-1")] + [os.path.join(os.getcwd(), ".\COVID19_1110\studies\CT-2", x)
    for x in os.listdir(".\COVID19_1110\studies\CT-2")] + [os.path.join(os.getcwd(), ".\COVID19_1110\studies\CT-3", x)
    for x in os.listdir(".\COVID19_1110\studies\CT-3")] + [os.path.join(os.getcwd(), ".\COVID19_1110\studies\CT-4", x)
    for x in os.listdir(".\COVID19_1110\studies\CT-4")
]

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths[1:25]])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths[1:25]])

print(normal_scans.shape)
print(abnormal_scans.shape)

# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

print(len(abnormal_scans))
print(len(normal_scans))

Percent_abn = len(abnormal_scans)*80//100
Percent_n = len(normal_scans)*80//100

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((abnormal_scans[:Percent_abn], normal_scans[:Percent_n]), axis=0)
y_train = np.concatenate((abnormal_labels[:Percent_abn], normal_labels[:Percent_n]), axis=0)
x_val = np.concatenate((abnormal_scans[Percent_abn:], normal_scans[Percent_n:]), axis=0)
y_val = np.concatenate((abnormal_labels[Percent_abn:], normal_labels[Percent_n:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

# Define data loaders.
#train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

#print(train_loader.shape)
#print(validation_loader.shape)

batch_size = 1

'''
model = models.Sequential([
  layers.Conv3D(filters=64, kernel_size=3, activation="relu", input_shape=(128, 128, 64, 1)),
  layers.MaxPool3D(pool_size=2),
  layers.BatchNormalization(),
  layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
  layers.MaxPool3D(pool_size=2),
  layers.BatchNormalization(),
  layers.Conv3D(filters=128, kernel_size=3, activation="relu"),
  layers.MaxPool3D(pool_size=2),
  layers.BatchNormalization(),
  layers.Conv3D(filters=256, kernel_size=3, activation="relu"),
  layers.MaxPool3D(pool_size=2),
  layers.BatchNormalization(),
  layers.Flatten(),
  layers.GlobalAveragePooling3D(),
  layers.Dense(units=512, activation="relu"),
  layers.Dropout(0.3),
  layers.Dense(128, activation='relu'),
  layers.Dense(2, activation='softmax')
])
'''

model = models.Sequential([
  layers.Conv2D(16, 4, padding='same', activation='relu', input_shape=(128, 128, 64)), # Convolitional layer with use of 3x3 filters # Padding so there is no data loss on the edge ( same = padding)
  layers.MaxPooling2D(),                                                             # Reduce dimensions of data together to reduce computation
  layers.Dropout(0.5),
  layers.Conv2D(16, 4, padding='same', activation='relu'),                           # ReLU (rectified linear activation function) is almost linear (can bend to approx. data)
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),                                                               # Randomly drops out 50% of output
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Flatten(),                                                                  # Flatten tensor to 1D
  layers.Dense(128, activation='relu'),                                              # Each neuron of this layer gets an input from each neuron of previous layer
  layers.Dense(2, activation='softmax')                                    # Number of output classes
])

epochs = 100
model.summary()

model.compile(optimizer='adam',                                                     # Gradient descent method
              loss='sparse_categorical_crossentropy', # Computes the crossentropy loss between the labels and predictions
              metrics=['accuracy'])

# Train model
history = model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_val, y_val))

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
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('Loss_acc.png')
plt.show()