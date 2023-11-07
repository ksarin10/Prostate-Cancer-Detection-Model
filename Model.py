'''
This file takes the patches from the resizing.py file and trains the model using those patches. The model 
is created using a U-Net architecture which is optimal for image segmentation processes. Before I train the
model I read all of the images using cv2 and put them through stain normalization, which is a technique 
to make all of the biopsies the same color and have the model produce the optimal results. After training
the model, I created a few functions to create all of the mask predictions and determine the correct 
coordinates for all of the masks. Then I created an empty mask that is the same size as the real mask 
from my dataset and I inserted all  of the masks in to their correct coordinates to get the final 
cancer mask.
'''



import os
import warnings
import random
import numpy as np
import math
import re
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
sys.path.append("/Users/krishsarin/Downloads/Krish/")
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import (
    Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization
)
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from normalization import stain_normalization
from scipy.linalg import LinAlgError



seed = 42
np.random.seed(seed)

IMG_WIDTH = 512
IMG_HEIGHT = 512

IMG_CHANNELS = 3


def lr_scheduler(epoch):
    initial_learning_rate = 0.001
    drop = 0.5  # Learning rate will be reduced by half every few epochs
    epochs_drop = 10  # Reduce learning rate every 10 epochs
    learning_rate = initial_learning_rate * (drop ** (epoch // epochs_drop))
    return learning_rate

# If lengths are not the same this can find differences
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

def get_image_paths_in_directory(directory):
    image_paths = []

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Checks if the file is a regular file (not a directory)
        if os.path.isfile(filepath):
            # Checks if the file is an image 
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            if any(filepath.lower().endswith(ext) for ext in valid_extensions):
                image_paths.append(filepath)

    return image_paths


train_image_dir = '/Users/krishsarin/Downloads/Krish/Training/img2c'
train_mask_dir = '/Users/krishsarin/Downloads/Krish/Training/mask2c'
val_image_dir = '/Users/krishsarin/Downloads/Krish/Validation/img2c'
val_mask_dir = '/Users/krishsarin/Downloads/Krish/Validation/mask2c'




train_image_paths = list_files_in_directory(train_image_dir)
print(len(train_image_paths))

normal_list = []

for image_path in train_image_paths:
    image = cv2.imread(image_path)
    normalized_image = stain_norm(image)
    normalized_image = np.array(normalized_image)
    normal_list.append(normalized_image)

img = normal_list[160]
plt.imshow(img)


train_mask_paths = list_files_in_directory(train_mask_dir)

mask_list = []

print(len(train_mask_paths))
for image_path in train_mask_paths:
    image = cv2.imread(image_path)
    image = np.array(image)
    mask_list.append(image)

val_image_paths = list_files_in_directory(val_image_dir)
len(val_image_paths)
print(val_image_paths)

normal_val_list = []
for patch in val_image_paths:
    image = cv2.imread(patch)
    normalized_image = stain_norm(image)
    normal_val_list.append(normalized_image)

img1 = normal_val_list[100]
plt.imshow(img1)

print(len(normal_val_list))

mask_val_list = []

val_mask_paths = list_files_in_directory(val_mask_dir)
for image_path in val_mask_paths:
    image = cv2.imread(image_path)
    image = np.array(image)
    mask_val_list.append(image)


print(len(val_mask_paths))



# Load image paths and mask paths
train_image_paths = list_files_in_directory(train_image_dir)
train_mask_paths = list_files_in_directory(train_mask_dir)
val_image_paths = list_files_in_directory(val_image_dir)
val_mask_paths = list_files_in_directory(val_mask_dir)
print(len(train_mask_paths))

# Shuffle the data
train_data = list(zip(train_image_paths, train_mask_paths))
val_data = list(zip(val_image_paths, val_mask_paths))
random.shuffle(train_data)
random.shuffle(val_data)

# Define stain-normalization function
def apply_stain_norm(image_path):
    image = cv2.imread(image_path)
    normalized_image = stain_norm(image)
    return normalized_image

def load_mask(image_path):
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return mask

# Apply stain normalization to the images
train_images = [apply_stain_norm(image_path) for image_path, _ in train_data]
val_images = [apply_stain_norm(image_path) for image_path, _ in val_data]
train_masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for _, mask_path in train_data]
val_masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for _, mask_path in val_data]

# Convert to NumPy arrays
train_images = np.array(train_images)
val_images = np.array(val_images)
train_masks = np.array(train_masks)
val_masks = np.array(val_masks)

print(len(train_masks))
plt.imshow(train_images[23])
plt.imshow(train_masks[23])

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    vertical_flip=True,
)
mask_datagen = ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = True,
)

train_masks = np.expand_dims(train_masks, axis=-1)

# Data generators for training and validation
batch_size = 16
train_data_generator = train_datagen.flow(train_images, train_masks, batch_size=batch_size, seed = seed)



val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only apply normalization

# Specify the same input size as your model's expected input size
input_size = (512, 512)

# Data generator for validation data
val_data_generator = val_datagen.flow(val_images, val_masks, batch_size=batch_size)

class CustomImageDataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, image_datagen, validation=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_datagen = image_datagen
        self.validation = validation

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
    
        X = np.zeros((len(batch_image_paths), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        y = np.zeros((len(batch_mask_paths), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    
        for i in range(len(batch_image_paths)):
            X[i] = cv2.imread(batch_image_paths[i]) / 255.0  # Load and normalize images
            mask = cv2.imread(batch_mask_paths[i], cv2.IMREAD_GRAYSCALE)  # Load masks as grayscale
            y[i] = mask[:, :, np.newaxis]  # Expand mask dimensions to (512, 512, 1)
    
        if not self.validation:  # Apply data augmentation only during training
            image_augmented = self.image_datagen.flow(X, shuffle=False)
            mask_augmented = self.mask_datagen.flow(y, shuffle=False)
            X = image_augmented[0]
            y = mask_augmented[0]
    
        return X, y

# Initialize data generators
train_data_generator = CustomImageDataGenerator(train_image_paths, train_mask_paths, batch_size, train_datagen, mask_datagen)
val_data_generator = CustomImageDataGenerator(val_image_paths, val_mask_paths, batch_size, None, validation=True)

num_samples_to_visualize = 9# You can change this number

for i in range(num_samples_to_visualize):
    sample = train_data_generator[i]
    images, masks = sample
    plt.figure(figsize=(12, 6))
    
    for j in range(batch_size):
        plt.subplot(num_samples_to_visualize, batch_size, j + 1)
        plt.imshow(images[j])
        plt.title('Sample Image')
        plt.axis('off')
        
        plt.subplot(num_samples_to_visualize, batch_size, j + batch_size + 1)
        plt.imshow(masks[j].squeeze(), cmap='gray')
        plt.title('Sample Mask')
        plt.axis('off')

plt.show()

lr_callback = LearningRateScheduler(lr_scheduler)

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255)(inputs)

# Contraction path
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = BatchNormalization()(c1)  
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
c1 = BatchNormalization()(c1)  
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = BatchNormalization()(c2)  
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
c2 = BatchNormalization()(c2)  
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = BatchNormalization()(c3) 
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
c3 = BatchNormalization()(c3)  
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = BatchNormalization()(c4)  
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
c4 = BatchNormalization()(c4)  
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = BatchNormalization()(c5) 
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
c5 = BatchNormalization()(c5) 

# Expansive path
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = BatchNormalization()(c6)  
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
c6 = BatchNormalization()(c6)  

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = BatchNormalization()(c7) 
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
c7 = BatchNormalization()(c7)  

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = BatchNormalization()(c8)  
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
c8 = BatchNormalization()(c8)  

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = BatchNormalization()(c9)  
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
c9 = BatchNormalization()(c9) 

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

# Create the model
model = Model(inputs=[inputs], outputs=[outputs])


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


#model summary
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

steps_per_epoch = math.ceil(len(train_image_paths) / batch_size)
validation_steps = math.ceil(len(val_image_paths) / batch_size)

history = model.fit(
    train_data_generator,
    epochs=100,
    validation_data=val_data_generator,
    callbacks=[lr_callback],
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)



model.summary()


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


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



img_path = "/Users/krishsarin/Downloads/Krish/new/512_img/Validation/120330/image_patch_695.png"

image = cv2.imread(img_path)

patch = np.array(image).astype(np.float32)

normalized_image = stain_norm(patch)

normalized_image = normalized_image / 255.0


input_data = np.expand_dims(normalized_image, axis=0)


single_patch_prediction = (model.predict(input_data)[0,:,:,0] > 0.5).astype(np.uint8)

plt.imshow(single_patch_prediction)

# File of real mask
real_mask_path = "/Users/krishsarin/Downloads/Krish/Testing/mask/120342/mask_patch_412.png"  


real_mask = Image.open(real_mask_path).convert("L")


real_mask_array = np.array(real_mask)

real_mask_array = (real_mask_array > 0).astype(np.uint8)

predicted_mask_flat = single_patch_prediction.flatten()
real_mask_flat = real_mask_array.flatten()

# Calculate the Dice coefficient
dice_coefficient = f1_score(real_mask_flat, predicted_mask_flat, average='binary')

print(f"Dice Coefficient: {dice_coefficient}")


image_directory = '/Users/krishsarin/Downloads/Krish/Testing/img/120342'
mask_directory = '/Users/krishsarin/Downloads/Krish/Testing/mask/120342'

dice_coefficients = []

for filename in os.listdir(image_directory):
    if filename.endswith(".png"):
        img_path = os.path.join(image_directory, filename)
        image = cv2.imread(img_path)
        patch = np.array(image).astype(np.float32)

        try:
            img_normalized = stain_normalization(image)
            img_array /= 255.0
        except LinAlgError:
            print("Stain normalization failed. Skipping normalization.")
            img = image.load_img(img_path, target_size=(512, 512))
            img_array = image.img_to_array(img)
            img_array /= 255.0

        input_data = np.expand_dims(img_array, axis=0)
        single_patch_prediction = (model2.predict(input_data)[0, :, :, 0] > 0.5).astype(np.uint8)
        image_number = filename.split("_")[-1].split(".")[0]
        mask_filename = f"mask_patch_{image_number}.png"
        mask_path = os.path.join(mask_directory, mask_filename)

        try:
            real_mask = Image.open(mask_path).convert("L")
            real_mask_array = np.array(real_mask)
            real_mask_array = (real_mask_array > 0).astype(np.uint8)
        except FileNotFoundError:
            print(f"Warning: Real mask file not found for {filename}. Skipping.")
            continue
        predicted_mask_flat = single_patch_prediction.flatten()
        real_mask_flat = real_mask_array.flatten()

        # Calculate the Dice coefficient
        dice_coefficient = f1_score(real_mask_flat, predicted_mask_flat, average='binary')
        dice_coefficients.append(dice_coefficient)


# Print the Dice coefficients for all images
average_dice = sum(dice_coefficients) / len(dice_coefficients)
print(f"Average Dice Coefficient: {average_dice}")






def create_full_mask_from_biopsy(biopsy_path, model):
    biopsy_image = Image.open(biopsy_path)  # Use OpenCV to read the biopsy image

    # Get the dimensions of the biopsy image
    biopsy_width, biopsy_height = biopsy_image.size
    print(f"Dimensions of biopsy_image: ({biopsy_width}, {biopsy_height})")

    patch_size = 256

    coordinates_list = []
    all_patches = []

    index = 0

    for y in range(0, biopsy_height, patch_size):
        for x in range(0, biopsy_width, patch_size):
            # Define the coordinates for the current patch
            patch_coords = (x, y, x + patch_size, y + patch_size)

            # Crop the patch from the biopsy image using Image.open
            patch = biopsy_image.crop(patch_coords)
            patch = np.array(patch)
            try:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                patch = stain_norm(patch)
            except:
                continue

            coordinates_list.append(patch_coords)
            all_patches.append(patch)

            index += 1
            
    tf.get_logger().setLevel('ERROR')

    # Parallel processing
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    batch_size = 32  
    batches = [all_patches[i:i+batch_size] for i in range(0, len(all_patches), batch_size)]

    all_masks = []

    for batch in batches:
        batch = np.array(batch) / 255.0
        input_data = np.array(batch)
        batch_predictions = model.predict(input_data)
        for patch_prediction in batch_predictions:
            mask = (patch_prediction[:, :, 0] > 0.5).astype(np.uint8)
            all_masks.append(mask)
            
    full_mask = np.zeros((biopsy_height, biopsy_width), dtype=np.uint8)

    pattern = r'(\d+),(\d+)'

    for mask, (left, upper, right, lower) in zip(all_masks, coordinates_list):
        if mask is None:
            print(f"Skipping None mask at coordinates ({left}, {upper}, {right}, {lower})")
            continue

        mask = mask * 255

        try:
            full_mask[upper:lower, left:right] = np.maximum(full_mask[upper:lower, left:right], mask)
        except ValueError as e:
            
            error_message = str(e)

            match = re.search(pattern, error_message)
            if match:
                expected_height, expected_width = map(int, match.groups())
            else:
                print(f"Failed to extract expected dimensions from the error message.")
                continue

            adjusted_upper = upper
            adjusted_lower = lower + (256 - expected_height)
            adjusted_left = left
            adjusted_right = right + (256 - expected_width)

            scale_x = expected_width / mask.shape[1]
            scale_y = expected_height / mask.shape[0]
          
            if scale_x < 1 and scale_y < 1:
                mask = cv2.resize(mask, (0, 0), fx=scale_x, fy=scale_y)
                full_mask[adjusted_upper:adjusted_lower, adjusted_left:adjusted_right] = np.maximum(full_mask[adjusted_upper:adjusted_lower, adjusted_left:adjusted_right], mask)
            
            full_mask_pil = Image.fromarray(full_mask)
            plt.imshow(full_mask_pil)
    

create_full_mask_from_biopsy("/Users/krishsarin/Downloads/Krish/level1/120366_level_1_image.png", model)


# Calculate Dice for a Singular Full mask

full_mask_path = "/Users/krishsarin/Downloads/Krish/Results/new1_120330.png"
full_mask_pil = Image.open(full_mask_path).convert("L")
full_mask_array = np.array(full_mask_pil)
full_mask_array = (full_mask_array > 0).astype(np.uint8)

real_mask_path = "/Users/krishsarin/Downloads/Krish/level1/120330_level_1_mask.png"
real_mask = Image.open(real_mask_path).convert("L")
real_mask_array = np.array(real_mask)
real_mask_array = (real_mask_array > 0).astype(np.uint8)


full_mask_flat = full_mask_array.flatten()
real_mask_flat = real_mask_array.flatten()

dice_coefficient = f1_score(real_mask_flat, full_mask_flat, average=None)

print("Real Mask Values:")
print(real_mask_array)

print("\nFull Mask Values:")
print(full_mask_array)


plt.imshow(real_mask_array, cmap='gray')
plt.title('Ground Truth Mask')
plt.show()

plt.imshow(full_mask_array, cmap='gray')
plt.title('Predicted Mask')
plt.show()

dice_coefficient_positive = dice_coefficient[1]
print(f"Dice Coefficient for Positive Class: {dice_coefficient_positive}")


