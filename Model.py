import os
import warnings
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from openslide import OpenSlide
from PIL import Image
import matplotlib.pyplot as plt
import atexit
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import signal
import sys
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import (
    Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization
)



seed = 42
np.random.seed(seed)

IMG_WIDTH = 512
IMG_HEIGHT = 512

IMG_CHANNELS = 3


train_image_dir = '/Users/krishsarin/Downloads/Krish/Training/img1'
train_mask_dir = '/Users/krishsarin/Downloads/Krish/Training/mask1'
val_image_dir = '/Users/krishsarin/Downloads/Krish/Validation/img1'
val_mask_dir = '/Users/krishsarin/Downloads/Krish/Validation/mask1'


def lr_schedule(epoch):
    initial_lr = 0.001  
    if epoch < 10:
        return initial_lr
    else:
        return initial_lr * tf.math.exp(0.1 * (10 - epoch))
    

# Dice loss function alternative
'''
def dice_coefficient(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + 1e-5) / (union + 1e-5)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)
'''

def list_files_in_directory(directory):
    file_paths = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            subdir_files = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path) if not file.startswith('.')]
            subdir_files = sorted(subdir_files)
            file_paths.extend(subdir_files)
    return file_paths


# Make sure lengths are the same for mask and images
train_image_paths = list_files_in_directory(train_image_dir)
print(len(train_image_paths))
train_mask_paths = list_files_in_directory(train_mask_dir)
print(len(train_mask_paths))
val_image_paths = list_files_in_directory(val_image_dir)
print(len(val_image_paths))
val_mask_paths = list_files_in_directory(val_mask_dir)
print(len(val_mask_paths))


# If lengths are not the same this can find differences
'''
def count_images_in_directory(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                count += 1
    return count

subdirectory_counts = {}
for subdirectory in os.listdir(val_mask_dir):
    subdirectory_path = os.path.join(val_mask_dir, subdirectory)
    if os.path.isdir(subdirectory_path):
        count = count_images_in_directory(subdirectory_path)
        subdirectory_counts[subdirectory] = count
    
for subdirectory, count in subdirectory_counts.items():
    print(f"Subdirectory '{subdirectory}' contains {count} images.")





assert len(train_image_paths) == len(train_mask_paths)
assert len(val_image_paths) == len(val_mask_paths)

print(len(train_image_paths))
'''

train_data = list(zip(train_image_paths, train_mask_paths))
val_data = list(zip(val_image_paths, val_mask_paths))

random.shuffle(train_data)
random.shuffle(val_data)



print(len(train_data))

print(len(val_data))


train_images = []
train_masks = []
val_images = []
val_masks = []

# Loop for adding training images to lists to convert to numpy array
for image_path, mask_path in zip(train_image_paths, train_mask_paths):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  
    
    
    if image is None or mask is None:
        print(f"Failed to load image {image_path} or mask {mask_path}.")
        continue
    
    if image.shape[:2] != (IMG_HEIGHT, IMG_WIDTH) or mask.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
        print(f"Image {image_path} or mask {mask_path} has incorrect dimensions.")
    
    train_images.append(image)
    train_masks.append(mask)

print(len(train_images))


# Loop for adding validation images to lists to convert to numpy array
for image_path, mask_path in zip(val_image_paths, val_mask_paths):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  
    
    
    if image is None or mask is None:
        print(f"Failed to load image {image_path} or mask {mask_path}.")
        continue  
    
    if image.shape[:2] != (IMG_HEIGHT, IMG_WIDTH) or mask.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
        print(f"Image {image_path} or mask {mask_path} has incorrect dimensions.")
    
    val_images.append(image)
    val_masks.append(mask)

    
width, height = train_images[1].size
print(width, height)
    
train_images = np.array(train_images)
train_masks = np.array(train_masks)
val_images = np.array(val_images)
val_masks = np.array(val_masks)

train_images = train_images / 255.0
train_masks = train_masks / 255.0
val_images = val_images / 255.0
val_masks = val_masks / 255.0



train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)



val_datagen = ImageDataGenerator(rescale=1.0 / 255)

batch_size = 16

# Create data generator with the augmentations
def custom_data_generator(image_paths, mask_paths, datagen, batch_size):
    image_generator = datagen.flow_from_directory(
        image_paths,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        seed=42  
    )

    mask_generator = datagen.flow_from_directory(
        mask_paths,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        seed=42  
    )

    data_generator = zip(image_generator, mask_generator)
    return data_generator

# Data generators
train_data_generator = custom_data_generator(train_image_dir, train_mask_dir, train_datagen, batch_size)
val_data_generator = custom_data_generator(val_image_dir, val_mask_dir, val_datagen, batch_size)



'''
def visualize_random_image_and_mask(images, masks, num_samples=50):
    num_validation_samples = len(images)  # Number of validation samples

    if num_samples > num_validation_samples:
        num_samples = num_validation_samples  # Limit num_samples to available data

    unique_indices = random.sample(range(num_validation_samples), num_samples)

    if num_samples == 0:
        print("No validation samples to visualize.")
        return

    for i, index in enumerate(unique_indices):
        image = images[index]
        mask = masks[index]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Validation Image {i + 1}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {i + 1}')
        plt.axis('off')

        plt.show()
'''

#visualization functions to make sure the data matches
def visualize_training_random_image_and_mask(images, masks, num_samples=50, dataset_type="Training"):
    num_samples = min(num_samples, len(train_data))  # Ensure num_samples doesn't exceed the available data

    if num_samples == 0:
        print(f"No {dataset_type} samples to visualize.")
        return

    unique_indices = random.sample(range(len(images)), num_samples)

    for i, index in enumerate(unique_indices):
        image = images[index]
        mask = masks[index]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'{dataset_type} Image {i + 1}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {i + 1}')
        plt.axis('off')

        plt.show()
        
def visualize_batch_from_generator(data_generator, num_samples=50):
    for i, batch in enumerate(data_generator):
        if i * batch_size >= num_samples:
            break
        
        images, masks = batch

        for j in range(len(images)):
            image = images[j]
            mask = masks[j]

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f'Image {i * batch_size + j + 1}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask {i * batch_size + j + 1}')
            plt.axis('off')

            plt.show()


train_data_generator = custom_data_generator(train_image_dir, train_mask_dir, train_datagen, batch_size)

#Visualization of images
visualize_batch_from_generator(train_data_generator, num_samples=2)  

visualize_random_image_and_mask(val_images, val_masks, num_samples=50)

visualize_training_random_image_and_mask(train_images, train_masks, num_samples=50, dataset_type="Training")

print(len(train_images))
print(len(train_masks))

    

# Input layer
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255)(inputs)

# Contraction path
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = BatchNormalization()(c1)  
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = BatchNormalization()(c2) 
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = BatchNormalization()(c3)  
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = BatchNormalization()(c4)  
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = BatchNormalization()(c5)  
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = BatchNormalization()(c6)  
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = BatchNormalization()(c7)  
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = BatchNormalization()(c8)  
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = BatchNormalization()(c9)  
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

# Build/compile the model
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#model summary
model.summary()

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

# Train the model
history = model.fit(
    train_data_generator,
    epochs=50,
    validation_data=val_data_generator,
    callbacks=[lr_scheduler, early_stopping, tensorboard],
    steps_per_epoch=len(train_image_paths) // batch_size,
    validation_steps=len(val_image_paths) // batch_size
)



val_loss, val_accuracy = model.evaluate(val_data_generator, steps=len(val_image_paths) // batch_size)
print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")



model.summary()


# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(history.history['loss'], label='Training Loss', color='red')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  
ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('Training and Validation Loss and Accuracy')
plt.show()



# Load the image from the file path for testing
img_path = "/Users/krishsarin/Downloads/Krish/Training/img1/120378/image_patch_6.png"
img = image.load_img(img_path, target_size=(512, 512)) 

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Normalize the pixel values to [0, 1]
img_array /= 255.0

# Add a batch dimension
input_data = np.expand_dims(img_array, axis=0)

# Make predictions using your model
predictions = model.predict(input_data)

# Display the input image
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")

# Display the predicted mask
plt.subplot(1, 2, 2)
plt.imshow(predictions[0], cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.show()




