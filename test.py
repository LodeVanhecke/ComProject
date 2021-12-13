import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib
import random
from scipy import ndimage
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.python.keras import datasets, layers, models, losses

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    #print(type(scan))
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    #print(type(volume))
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
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

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

def rotate(volume):
    # define some rotation angles
    angles = [-80, -40, -20, -10, -5, 5, 10, 20, 40, 80]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, reshape=False)
    volume[volume < 0] = 0
    volume[volume > 1] = 1
    return volume

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

x_train = np.array([rotate(x) for x in x_train])

#data_augmentation = keras.Sequential([
#    layers.RandomFlip("horizontal", input_shape=(128, 128, 64),
#    layers.RandomRotation(0.1),
#    layers.RandomZoom(0.1),
#])

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
model = models.Sequential([
  layers.Conv2D(16, 4, padding='same', activation='relu', input_shape=(128, 128, 64)), # Convolitional layer with use of 4x4 filters # Padding so there is no data loss on the edge ( same = padding)
  layers.MaxPooling2D(),                                                               # Reduce dimensions of data together to reduce computation
  layers.Dropout(0.4),
  layers.Conv2D(16, 4, padding='same', activation='relu'),                             # ReLU (rectified linear activation function) is almost linear (can bend to approx. data)
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Conv2D(16, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),                                                                 # Randomly drops out 50% of output
  layers.Conv2D(16, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Conv2D(16, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Flatten(),                                                                  # Flatten tensor to 1D
  layers.Dense(128, activation='relu'),                                              # Each neuron of this layer gets an input from each neuron of previous layer
  layers.Dense(2, activation='softmax')                                              # Number of output classes
])
'''
batch_size = 2

epochs = 10
model.summary()

model.compile(optimizer='adam',                                                     # Gradient descent method
              loss='sparse_categorical_crossentropy',                               # Computes the crossentropy loss between the labels and predictions
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
plt.plot(epochs_range, loss, label='Training Looss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('Loss_acc.png')
plt.show()
'''
