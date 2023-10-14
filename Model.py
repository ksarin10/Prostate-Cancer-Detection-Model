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


train_image_dir = '/Users/krishsarin/Downloads/Krish/Training/img2c'
train_mask_dir = '/Users/krishsarin/Downloads/Krish/Training/mask2c'
val_image_dir = '/Users/krishsarin/Downloads/Krish/Validation/img2c'
val_mask_dir = '/Users/krishsarin/Downloads/Krish/Validation/mask2c'
sample_directory = '/Users/krishsarin/Downloads/Krish/Training/img2c/120345'


def lr_scheduler(epoch):
    initial_learning_rate = 0.001
    drop = 0.5  # Learning rate will be reduced by half every few epochs
    epochs_drop = 10  # Reduce learning rate every 10 epochs
    learning_rate = initial_learning_rate * (drop ** (epoch // epochs_drop))
    return learning_rate

    

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


directory_path = '/Users/krishsarin/Downloads/Krish/Training/img2c/120364'
image_list = get_image_paths_in_directory(directory_path)
print(image_list)




train_image_paths = list_files_in_directory(train_image_dir)
print(len(train_image_paths))

normal_list = []
for patch in train_image_paths:
    normalized_image = stain_normalization(patch)
    normal_list.append(normalized_image)

print(len(normal_list))

img = normal_list[0]
plt.imshow(img)

    
img = '/Users/krishsarin/Downloads/Krish/Testing/img/120342img/120342/image_patch_1747.png'
normal_img = stain_normalization(img)
plt.imshow(normal_img)


train_mask_paths = list_files_in_directory(train_mask_dir)
print(len(train_mask_paths))
mask_image = cv2.imread(train_mask_paths[0], cv2.IMREAD_UNCHANGED)
plt.imshow(mask_image)

val_image_paths = list_files_in_directory(val_image_dir)

normal_val_list = []
for patch in val_image_paths:
    normalized_image = stain_normalization(patch)
    normal_val_list.append(normalized_image)

img1 = normal_val_list[9]
plt.imshow(img1)

print(len(val_image_paths))
val_mask_paths = list_files_in_directory(val_mask_dir)
print(len(val_mask_paths))





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





assert len(normal_list) == len(train_mask_paths)
assert len(normal_val_list) == len(val_mask_paths)

print(len(train_image_paths))




train_data = list(zip(normal_list, train_mask_paths))
val_data = list(zip(normal_val_list, val_mask_paths))

random.shuffle(train_data)
random.shuffle(val_data)



print(len(train_data))

print(len(val_data))




def visualize_data(data, num_samples=5):
    # Randomly select `num_samples` samples from the data
    samples = random.sample(data, num_samples)

    # Create subplots for the samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))

    for i, (image, mask_path) in enumerate(samples):
        # Load the mask image
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Display the image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Sample {i + 1}: Image")
        axes[i, 0].axis("off")

        # Display the mask
        axes[i, 1].imshow(mask_image, cmap='gray')
        axes[i, 1].set_title(f"Sample {i + 1}: Mask")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

# Visualize a random sample from the training data
visualize_data(train_data)

# Visualize a random sample from the validation data
visualize_data(val_data)







val_images = []
val_masks = []
train_images = []  
train_masks = []   

for image_path, mask_path in zip(train_image_paths, train_mask_paths):
  
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    image_normalized = stain_normalization(image_path)

    # Check if the image and mask have the correct dimensions
    if image_normalized.shape[:2] != (IMG_HEIGHT, IMG_WIDTH) or mask.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
        print(f"Image {image_path} or mask {mask_path} has incorrect dimensions.")

    train_images.append(image_normalized)
    train_masks.append(mask)


# Loop for adding validation images to lists to convert to numpy array
for image_path, mask_path in zip(val_image_paths, val_mask_paths):
    # Load the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Apply stain normalization
    image_normalized = stain_normalization(image_path)

    # Check if the image and mask have the correct dimensions
    if image_normalized.shape[:2] != (IMG_HEIGHT, IMG_WIDTH) or mask.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
        print(f"Image {image_path} or mask {mask_path} has incorrect dimensions.")

    val_images.append(image_normalized)
    val_masks.append(mask)


def visualize_sample(image, mask):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Normalized Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    
    plt.show()

# Choose an index to visualize a specific training sample
sample_index = 70 # Change this to the index of the sample you want to visualize

# Visualize the sample
visualize_sample(train_images[sample_index], train_masks[sample_index])

visualize_sample(val_images[sample_index], val_masks[sample_index])






    
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
    num_samples = min(num_samples, len(train_data))  

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
        
def visualize_batch_from_generator(data_generator, num_samples=50, samples_per_batch=5):
    all_batches = list(data_generator)
    random.shuffle(all_batches)  # Shuffle the batches to get random samples

    displayed_samples = 0

    for batch in all_batches:
        if displayed_samples >= num_samples:
            break 

        images, masks = batch

        
        images = images / 255.0
        masks = masks / 255.0    

        num_samples_in_batch = images.shape[0]
        samples_to_visualize = min(samples_per_batch, num_samples - displayed_samples)

        for j in range(samples_to_visualize):
            image = images[j]
            mask = masks[j]

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f'Image {displayed_samples + 1}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask {displayed_samples + 1}')
            plt.axis('off')

            plt.show()

            displayed_samples += 1

train_data_generator = custom_data_generator(train_image_dir, train_mask_dir, train_datagen, batch_size)

#Visualization of images
visualize_batch_from_generator(train_data_generator, num_samples=2)  

visualize_random_image_and_mask(val_images, val_masks, num_samples=50)

visualize_training_random_image_and_mask(train_images, train_masks, num_samples=50, dataset_type="Training")

print(len(train_images))
print(len(train_masks))

lr_callback = LearningRateScheduler(lr_scheduler)

    

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255)(inputs)

# Contraction path
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = BatchNormalization()(c1)  # Add batch normalization
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
c1 = BatchNormalization()(c1)  # Add batch normalization
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = BatchNormalization()(c2)  # Add batch normalization
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
c2 = BatchNormalization()(c2)  # Add batch normalization
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = BatchNormalization()(c3)  # Add batch normalization
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
c3 = BatchNormalization()(c3)  # Add batch normalization
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = BatchNormalization()(c4)  # Add batch normalization
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
c4 = BatchNormalization()(c4)  # Add batch normalization
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = BatchNormalization()(c5)  # Add batch normalization
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
c5 = BatchNormalization()(c5)  # Add batch normalization

# Expansive path
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = BatchNormalization()(c6)  # Add batch normalization
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
c6 = BatchNormalization()(c6)  # Add batch normalization

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = BatchNormalization()(c7)  # Add batch normalization
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
c7 = BatchNormalization()(c7)  # Add batch normalization

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = BatchNormalization()(c8)  # Add batch normalization
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
c8 = BatchNormalization()(c8)  # Add batch normalization

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = BatchNormalization()(c9)  # Add batch normalization
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
c9 = BatchNormalization()(c9)  # Add batch normalization

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
    epochs=75,
    validation_data=val_data_generator,
    callbacks=[lr_callback],
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
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
img_path = "/Users/krishsarin/Downloads/Krish/Validation/img2c/120333/image_patch_799.png"
img = stain_normalization(img_path)
img = image.load_img(img_path, target_size=(512, 512)) 

# Convert the image to a numpy array
img_array = image.img_to_array(img)


img_array /= 255.0


input_data = np.expand_dims(img_array, axis=0)


single_patch_prediction = (model.predict(input_data)[0,:,:,0] > 0.5).astype(np.uint8)

plt.imshow(single_patch_prediction)

# Define the file path to the real mask
real_mask_path = "/Users/krishsarin/Downloads/Krish/Testing/mask/120342/mask_patch_412.png"  # Change this to the actual path

# Load the real mask as a grayscale image
real_mask = Image.open(real_mask_path).convert("L")

# Convert the real mask to a numpy array
real_mask_array = np.array(real_mask)

# Normalize the real mask if needed (e.g., convert it to binary)
# For binary masks, you can threshold it to 0 and 1 values.
real_mask_array = (real_mask_array > 0).astype(np.uint8)


predicted_mask_flat = single_patch_prediction.flatten()
real_mask_flat = real_mask_array.flatten()

# Calculate the Dice coefficient
dice_coefficient = f1_score(real_mask_flat, predicted_mask_flat, average='binary')

print(f"Dice Coefficient: {dice_coefficient}")


image_directory = "/Users/krishsarin/Downloads/Krish/Testing/img/120342"
mask_directory = "/Users/krishsarin/Downloads/Krish/Testing/mask/120342"



img1 = stain_normalization("/Users/krishsarin/Downloads/Krish/Validation/img2c/120333/image_patch_378.png")
plt.imshow(img1)
img_array = image.img_to_array(img1)
img_array /= 255.0
input_data = np.expand_dims(img_array, axis=0)
single_patch_prediction = (model.predict(img1)[0, :, :, 0] > 0.5).astype(np.uint8)
plt.imshow(single_patch_prediction)


dice_coefficients = []

# Loop through all image files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".png"):
        # Load the image
        img_path = os.path.join(image_directory, filename)

        try:
            # Attempt stain normalization
            img_normalized = stain_normalization(img_path)

            # Prepare the image for prediction
            img_array = image.img_to_array(img_normalized)
            img_array /= 255.0
        except LinAlgError:
            # Handle the LinAlgError (Eigenvalues did not converge)
            print("Stain normalization failed. Skipping normalization.")
            # Load the image without normalization
            img = image.load_img(img_path, target_size=(512, 512))
            img_array = image.img_to_array(img)
            img_array /= 255.0

        input_data = np.expand_dims(img_array, axis=0)

        # Make predictions
        single_patch_prediction = (model.predict(input_data)[0, :, :, 0] > 0.5).astype(np.uint8)

        # Extract the image number from the filename
        image_number = filename.split("_")[-1].split(".")[0]

        # Construct the path to the corresponding mask
        mask_filename = f"mask_patch_{image_number}.png"
        mask_path = os.path.join(mask_directory, mask_filename)

        try:
            # Load the real mask
            real_mask = Image.open(mask_path).convert("L")
            real_mask_array = np.array(real_mask)
            real_mask_array = (real_mask_array > 0).astype(np.uint8)
        except FileNotFoundError:
            print(f"Warning: Real mask file not found for {filename}. Skipping.")
            continue

        # Flatten the predicted and real masks
        predicted_mask_flat = single_patch_prediction.flatten()
        real_mask_flat = real_mask_array.flatten()

        # Calculate the Dice coefficient
        dice_coefficient = f1_score(real_mask_flat, predicted_mask_flat, average='binary')

        # Append the Dice coefficient to the list
        dice_coefficients.append(dice_coefficient)


# Print the Dice coefficients for all images
average_dice = sum(dice_coefficients) / len(dice_coefficients)
print(f"Average Dice Coefficient: {average_dice}")






print(f"Number of patches needed horizontally: {num_patches_horizontal}")
print(f"Number of patches needed vertically: {num_patches_vertical}")


# Output directories for saving patches
output_image_dir = "/Users/krishsarin/Downloads/Krish/Validation/img2c/120330img"
output_mask_dir = "/Users/krishsarin/Downloads/Krish/Validation/mask2c/120330mask"

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Patch size


image_path = "/Users/krishsarin/Downloads/Krish/level1/120342_level_1_image.png"
mask_path = image_path.replace("_image.png", "_mask.png")  # Assumes mask follows naming convention

# Open the image and mask
image = Image.open(image_path)
mask = Image.open(mask_path)

# Get image dimensions
image_width, image_height = image.size
print(image_width, image_height)
# Get the image number from the file name
image_number = os.path.basename(image_path).split("_")[0]

patch_size = 512
coordinates_list = []

# Loop through the image to create patches
index = 0
for y in range(0, image_height, patch_size):
    for x in range(0, image_width, patch_size):
        # Define the coordinates for the current patch
        patch_coords = (x, y, x + patch_size, y + patch_size)

        # Append the patch coordinates to the list
        coordinates_list.append(patch_coords)

        # Crop the patch from the image and mask
        image_patch = image.crop(patch_coords)
        mask_patch = mask.crop(patch_coords)

        # Create subdirectories for each image and mask pair
        image_subdir = os.path.join(output_image_dir, image_number)
        mask_subdir = os.path.join(output_mask_dir, image_number)
        
        os.makedirs(image_subdir, exist_ok=True)
        os.makedirs(mask_subdir, exist_ok=True)

        # Save the patch in the corresponding subdirectories
        image_patch.save(os.path.join(image_subdir, f"image_patch_{index}.png"), format="PNG")
        mask_patch.save(os.path.join(mask_subdir, f"mask_patch_{index}.png"), format="PNG")

        index += 1

print(len(coordinates_list))

mask_directory = "/Users/krishsarin/Downloads/Krish/Validation/mask2c/120333"



# Create a list to store the coordinates of cancer patches
cancer_patch_coordinates = []

# Create a dictionary to map patch indices to filenames
patch_indices = {}

# Iterate through the files in the mask directory
for mask_filename in os.listdir(mask_directory):
    if mask_filename.endswith(".png"):
        # Extract the patch index from the mask filename
        patch_index = int(mask_filename.split("_")[-1].split(".")[0])
        patch_indices[patch_index] = mask_filename

# Sort the patch indices numerically
sorted_patch_indices = sorted(patch_indices)

# Iterate through the sorted patch indices
for patch_index in sorted_patch_indices:
    if patch_index in range(len(coordinates_list)):
        # Append the patch coordinates to the cancer_patch_coordinates list
        cancer_patch_coordinates.append(coordinates_list[patch_index])

# Print the sorted cancer_patch_coordinates
print(cancer_patch_coordinates)
print(len(cancer_patch_coordinates))


cancer_patch_indices = []

# Initialize a counter for the cancerous patches
cancer_patch_count = 0

# Iterate through all patch positions
for i, (left, upper, right, lower) in enumerate(coordinates_list):
    # Check if the current coordinates are in cancer_patch_coordinates
    if (left, upper, right, lower) in cancer_patch_coordinates:
        cancer_patch_indices.append(i)
        cancer_patch_count += 1

print(f"Number of cancerous patches: {cancer_patch_count}")
print(f"Indices of cancerous patches: {cancer_patch_indices}")




image_directory = "/Users/krishsarin/Downloads/Krish/Testing/img/120342img"

cancer_img_directory = '/Users/krishsarin/Downloads/Krish/Testing/img/120342'


all_patches = []



# Define a regular expression pattern to extract the numeric part of the filename
pattern = r'image_patch_(\d+)\.png'

# Create a list to store the images
all_patches = [None] * len(coordinates_list)  # Initialize with None

for image_file in image_files:
    if image_file.endswith(".png"):
        image_path = os.path.join(image_directory, image_file)

        # Verify that the image file exists
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            continue  # Skip this image if it doesn't exist

        # Load the image
        image = cv2.imread(image_path)

        # Check if the image is not empty
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue  # Skip this image if it's empty

        # Check if it's a cancer patch
        is_cancer_patch = os.path.exists(os.path.join(cancer_img_directory, image_file))

        # Use regular expression to extract the patch index from the filename
        match = re.search(pattern, image_file)
        if match:
            patch_index = int(match.group(1))
            
            # Ensure that the patch_index is within the valid range
            if 0 <= patch_index < len(all_patches):
                # Apply stain normalization to cancer patches only
                if is_cancer_patch:
                    image = stain_normalization(image_path)
                    print(f"Normalized {image_path}")

                # Store the image in the correct index
                all_patches[patch_index] = image

print(len(all_patches))

input1 = '/Users/krishsarin/Downloads/Krish/Testing/img/120342/image_patch_212.png'
plt.imshow(stain_normalization(input1))

# Define the range of indices you want to visualize

plt.imshow(all_patches[213])

start_index = 210
end_index = 215

# Loop through and visualize the images within the specified range
for patch_index in range(start_index, end_index + 1):
    # Get the image from the specified index
    image = all_patches[patch_index]

    # Display the image
    plt.figure(figsize=(5, 5))
    plt.axis('off')  # Turn off the axis
    plt.title(f"Patch {patch_index}")
    plt.show()








mask_directory = "/Users/krishsarin/Downloads/Krish/Validation/mask2c/120330mask/120330"
cancer_mask_directory = '/Users/krishsarin/Downloads/Krish/Validation/mask2c/120330'







# Print the length of all_patches for debugging

print(f"Number of patches loaded: {len(all_patches)}")
for i, (image, mask) in enumerate(all_patches):
    if mask is not None:
        print(f"Patch {i}: Loaded successfully")
    else:
        print(f"Patch {i}: Mask is None")

plt.imshow(all_patches[213])


# Dice coefficient for singular patch
'''
for i, (image, mask) in enumerate(random_patches):
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)  # Display the image
    plt.title(f"Random Patch {i} - Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')  # Display the mask in grayscale
    plt.title(f"Random Patch {i} - Mask")
    
    # Display the path of the patch as text
    image_file = f"image_patch_{i}.png"  # Replace this with the correct way to get the image file name
    mask_file = image_file.replace("image_patch", "mask_patch")  # Adjust this if your mask filenames differ
    plt.text(10, 10, f"Image Path: {image_directory}/{image_file}", fontsize=10, color='red')
    plt.text(10, 30, f"Mask Path: {mask_directory}/{mask_file}", fontsize=10, color='blue')
    
    plt.show()  # Show the images

image_to_predict, _ = random_patches[5]

# Ensure that the image is in the same format as the one that worked
image_to_predict = image_to_predict / 255.0  # Normalize the image

# Prepare the image for prediction
input_data = np.expand_dims(image_to_predict, axis=0)

# Make a prediction for the selected image
single_patch_prediction = (model.predict(input_data)[0, :, :, 0] > 0.5).astype(np.uint8)
plt.imshow(single_patch_prediction)


# Define the file path to the real mask
real_mask_path = "/Users/krishsarin/Downloads/Krish/Testing/mask/120342/mask_patch_412.png"  # Change this to the actual path

# Load the real mask as a grayscale image
real_mask = Image.open(real_mask_path).convert("L")

# Convert the real mask to a numpy array
real_mask_array = np.array(real_mask)

# Normalize the real mask if needed (e.g., convert it to binary)
# For binary masks, you can threshold it to 0 and 1 values.
real_mask_array = (real_mask_array > 0).astype(np.uint8)


predicted_mask_flat = single_patch_prediction.flatten()
real_mask_flat = real_mask_array.flatten()

# Calculate the Dice coefficient
dice_coefficient = f1_score(real_mask_flat, predicted_mask_flat, average='binary')

print(f"Dice Coefficient: {dice_coefficient}")
'''

all_masks = []

for image in all_patches:
    # Normalize the image
    image = image / 255.0
    
    # Prepare the image for prediction
    input_data = np.expand_dims(image, axis=0)
    
    # Make predictions
    patch_prediction = (model.predict(input_data)[0, :, :, 0] > 0.5).astype(np.uint8)
    
    # Debugging: Print the shape of predictions
    print(f"Prediction shape for patch {len(all_masks)}: {patch_prediction.shape}")
    
    all_masks.append(patch_prediction)

start_index = 1746
end_index = 1748

for i in range(start_index, end_index + 1):
    plt.figure(figsize=(5, 5))
    plt.imshow(all_masks[i], cmap='gray')  # Assuming the masks are grayscale
    plt.axis('off')  # Turn off the axis
    plt.title(f"Mask {i}")
    plt.show()

print(f"Number of masks generated: {len(all_masks)}")

cmap = plt.get_cmap('jet')

# Number of masks to visualize
num_masks_to_visualize = 100

# Randomly select masks to visualize
random_masks = random.sample(all_masks, num_masks_to_visualize)

for i, mask in enumerate(random_masks):
    plt.figure(figsize=(6, 6))
    
    plt.imshow(mask, cmap='gray')  # Show the mask in black and white
    plt.title(f"Random Mask {i}")
    plt.show()

plt.imshow(random_masks[86])

image = Image.open('/Users/krishsarin/Downloads/Krish/level1/120342_level_1_image.png')

# Get the image width and height
width, height = image.size

print(f"Image Width: {width} pixels")
print(f"Image Height: {height} pixels")

full_mask = np.zeros((height, width), dtype=np.uint8)

for index, (left, upper, right, lower) in zip(cancer_patch_indices, cancer_patch_coordinates):
    print(f"Processing patch at coordinates: ({left}, {upper}, {right}, {lower})")

    # Ensure the mask is binary (0 and 255)
    mask = all_masks[index] * 255

    # Get the dimensions of the mask
    mask_height, mask_width = mask.shape

    print(f"Mask dimensions: height={mask_height}, width={mask_width}")

    if mask_height == 0:
        print(f"Skipping the current patch at coordinates: ({left}, {upper}, {right}, {lower}) due to unusual shape")
        continue

    # Calculate the region where the mask will be placed
    y_start, x_start = upper, left
    y_end, x_end = lower, right

    try:
        # Add the mask to the full mask at the specified coordinates
        full_mask[y_start:y_end, x_start:x_end] = np.maximum(full_mask[y_start:y_end, x_start:x_end], mask)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Skipping the current patch at coordinates: ({left}, {upper}, {right}, {lower})")
        continue




# Now full_mask contains the full mask with masks inserted at their respective coordinates
full_mask_pil = Image.fromarray(full_mask)
plt.imshow(full_mask_pil)
path =  Image.open('/Users/krishsarin/Downloads/Krish/level1/120342_level_1_mask.png')
plt.imshow(path)


# Load the mask image
mask = Image.open('/Users/krishsarin/Downloads/Krish/level1/120333_level_1_mask.png').convert("L")

# Get the coordinates for the first patch from cancer_patch_coordinates
patch_coordinates = cancer_patch_coordinates[2]
print(cancer_patch_indices[0])

# Extract the patch from the mask
patch_mask = mask.crop(patch_coordinates)

# Display the patch mask
plt.imshow(patch_mask, cmap='gray')
plt.title("Mask Patch")
plt.show()


# Display the original mask
plt.imshow(mask, cmap='gray')
plt.title("Original Mask")
plt.show()

# Display the full_mask
plt.imshow(full_mask_pil, cmap='gray')
plt.title("Full Mask")
plt.show()



first_patch = full_mask[cancer_patch_coordinates[0][1]:cancer_patch_coordinates[0][3], 
                        cancer_patch_coordinates[0][0]:cancer_patch_coordinates[0][2]]

# Display the first patch
plt.imshow(first_patch, cmap='gray')
plt.title("First Patch")
plt.show()

plt.imshow(all_masks[197])

print(cancer_patch_coordinates[0])


    
full_mask_path = "/Users/krishsarin/Downloads/Krish/new_img120342.png"

# Save the full mask as an image
full_mask_pil.save(full_mask_path)

print(f"Full mask saved to {full_mask_path}")


def create_patches(level_1_image_path, output_image_dir, patch_size=512):
    # Create output directory if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)

    # Open the level 1 image
    level_1_image = Image.open(level_1_image_path)

    # Get image dimensions
    image_width, image_height = level_1_image.size

    # Initialize the coordinates list
    coordinates_list = []

    index = 0

    for y in range(0, image_height, patch_size):
        for x in range(0, image_width, patch_size):
            # Define the coordinates for the current patch
            patch_coords = (x, y, x + patch_size, y + patch_size)

            # Append the patch coordinates to the list
            coordinates_list.append(patch_coords)

            # Crop the patch from the level 1 image
            patch = level_1_image.crop(patch_coords)

            # Save the patch as an image
            patch.save(os.path.join(output_image_dir, f"image_patch_{index}.png"), format="PNG")

            index += 1

    return coordinates_list

# Example usage:
level_1_image_path = "/Users/krishsarin/Downloads/Krish/level1/120330_level_1_image.png"
output_image_dir = "/Users/krishsarin/Downloads/Krish/Validation/img2c/120330img"
coordinates_list = create_patches(level_1_image_path, output_image_dir)
print(len(coordinates_list))



def create_full_mask(image_directory, mask_directory, coordinates_list, model, output_path, full_image):
    # Initialize lists for cancer patch information
    cancer_patch_indices = []
    cancer_patch_coordinates = []

    # Create a dictionary to map patch indices to filenames
    patch_indices = {}

    # Iterate through the files in the mask directory
    for mask_filename in os.listdir(mask_directory):
        if mask_filename.endswith(".png"):
            # Extract the patch index from the mask filename
            patch_index = int(re.search(r"mask_patch_(\d+)\.png", mask_filename).group(1))
            patch_indices[patch_index] = mask_filename

    # Sort the patch indices numerically
    sorted_patch_indices = sorted(patch_indices)

    # Iterate through the sorted patch indices
    for patch_index in sorted_patch_indices:
        if patch_index in range(len(coordinates_list)):
            # Append the patch index to cancer_patch_indices
            cancer_patch_indices.append(patch_index)
            # Append the patch coordinates to cancer_patch_coordinates
            cancer_patch_coordinates.append(coordinates_list[patch_index])

    # Check the length of cancer_patch_coordinates
    if len(cancer_patch_coordinates) == len(coordinates_list):
        raise ValueError("The length of cancer_patch_coordinates matches the length of coordinates_list. There may be an issue.")

    # After building cancer_patch_indices and cancer_patch_coordinates
    print(f"Length of cancer_patch_indices: {len(cancer_patch_indices)}")
    print(f"Length of cancer_patch_coordinates: {len(cancer_patch_coordinates)}")

    # Create a list to store the images
    all_patches = [None] * len(coordinates_list)

    image_files = [f for f in os.listdir(image_directory) if f.endswith(".png")]

    for image_file in image_files:
        if image_file.endswith(".png"):
            image_path = os.path.join(image_directory, image_file)

            # Check if it's a cancer patch
            is_cancer_patch = os.path.exists(os.path.join(mask_directory, image_file.replace("image", "mask")))

            # Use regular expression to extract the patch index from the filename
            match = re.search(r"image_patch_(\d+)\.png", image_file)
            if match:
                patch_index = int(match.group(1))

                # Ensure that the patch_index is within the valid range
                if 0 <= patch_index < len(all_patches):
                    # Load the image
                    image = cv2.imread(image_path)

                    # Apply stain normalization to cancer patches only
                    if is_cancer_patch:
                        # Apply stain normalization
                        # Replace the following line with your stain normalization logic
                        image = stain_normalization(image_path)

                    # Store the image in the correct index
                    all_patches[patch_index] = image

    all_masks = []

    for image in all_patches:
        if image is not None:
            # Normalize the image
            image = image / 255.0

            # Prepare the image for prediction
            input_data = np.expand_dims(image, axis=0)

            # Make predictions
            patch_prediction = (model.predict(input_data)[0, :, :, 0] > 0.5).astype(np.uint8)

            all_masks.append(patch_prediction)
    
    print("Length of all_masks:", len(all_masks))

    # Get image dimensions from the provided full image
    full_image = Image.open(full_image)
    image_width, image_height = full_image.size

    # Initialize the full mask with zeros
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Iterate through the cancer patch indices and their corresponding coordinates
    for index, (left, upper, right, lower) in zip(cancer_patch_indices, cancer_patch_coordinates):
        mask = all_masks[index]
        if mask.size == 0:
            print(f"Empty mask at index {index} for coordinates ({left}, {upper}, {right}, {lower})")
            continue
        
        mask = mask * 255  # Ensure binary (0 and 255)
    
        mask_height, mask_width = mask.shape

        y_start, x_start = upper, left
        y_end, x_end = lower, right

        if y_end - y_start != mask_height or x_end - x_start != mask_width:
            print(f"Shape mismatch for mask at index {index} and coordinates ({left}, {upper}, {right}, {lower})")
            continue
        
        full_mask[y_start:y_end, x_start:x_end] = np.maximum(full_mask[y_start:y_end, x_start:x_end], mask)

    # Convert the full_mask to a PIL image
    full_mask_pil = Image.fromarray(full_mask)

    # Save the full mask as an image
    full_mask_pil.save(output_path)

    print(f"Full mask saved to {output_path}")

# Example usage:
image_directory = "/Users/krishsarin/Downloads/Krish/Validation/img2c/120330img"
mask_directory = "/Users/krishsarin/Downloads/Krish/Validation/mask2c/120330"
model = model  
output_path = "/Users/krishsarin/Downloads/Krish/new_120330.png"
full_image_path = "/Users/krishsarin/Downloads/Krish/level1/120330_level_1_mask.png"  # Provide the full image path
create_full_mask(image_directory, mask_directory, coordinates_list, model, output_path, full_image_path)



#Calculate dice coefficient for singular full mask


image_path = "/Users/krishsarin/Downloads/Krish/level1/120333_level_1_mask.png"
image = Image.open(image_path)

# Get the dimensions (width and height) of the image
image_width, image_height = image.size

# Print the dimensions
print(f"Image width: {image_width} pixels")
print(f"Image height: {image_height} pixels")

# Load the full mask image from the given path
full_mask_path = "/Users/krishsarin/Downloads/Krish/Results/new_120330.png"
full_mask_pil = Image.open(full_mask_path).convert("L")
full_mask_array = np.array(full_mask_pil)
full_mask_array = (full_mask_array > 0).astype(np.uint8)

# Load the real mask from the given path
real_mask_path = "/Users/krishsarin/Downloads/Krish/level1/120330_level_1_mask.png"
real_mask = Image.open(real_mask_path).convert("L")
real_mask_array = np.array(real_mask)
real_mask_array = (real_mask_array > 0).astype(np.uint8)

# Flatten the masks
full_mask_flat = full_mask_array.flatten()
real_mask_flat = real_mask_array.flatten()

# Calculate the Dice coefficient
dice_coefficient = f1_score(real_mask_flat, full_mask_flat, average=None)

print("Real Mask Values:")
print(real_mask_array)

print("\nFull Mask Values:")
print(full_mask_array)


plt.imshow(real_mask_array, cmap='gray')
plt.title('Ground Truth Mask')
plt.show()

# Visualize full_mask
plt.imshow(full_mask_array, cmap='gray')
plt.title('Predicted Mask')
plt.show()

dice_coefficient_positive = dice_coefficient[1]
print(f"Dice Coefficient for Positive Class: {dice_coefficient_positive}")





# Define the dimensions
wsi_width = 136144
wsi_height = 60207
wsi_height_level1 = 15051  # Height of the level 1 slide
wsi_width_level1 = 34036   # Width of the level 1 slide

patch_width = 512
patch_height = 512

num_actual_patches = len(all_masks)
print(num_actual_patches)

# Calculate the number of patches needed in each direction based on the patches you have
num_patches_horizontal = wsi_width_level1 // patch_width
num_patches_vertical = wsi_height_level1 // patch_height

# Calculate the gap between patches
horizontal_gap = (wsi_width_level1 - (num_patches_horizontal * patch_width)) // (num_patches_horizontal - 1)
print(horizontal_gap)
vertical_gap = (wsi_height_level1 - (num_patches_vertical * patch_height)) // (num_patches_vertical - 1)
print(vertical_gap)

patch_positions = []
row = 0
col = 0

while len(patch_positions) < num_actual_patches:
    # Calculate the position based on patch dimensions and gap
    x = col * (patch_width + horizontal_gap)
    y = row * (patch_height + vertical_gap)
    patch_positions.append((y, x))  # Append (row, col) positions
    
    col += 1
    
    if col >= num_patches_horizontal:
        col = 0
        row += 1
        
print(len(patch_positions))

patch_size = 512
coordinates_list = []

# Loop through the image to create patches
index = 0
for y in range(0, image_height, patch_size):
    for x in range(0, image_width, patch_size):
        # Define the coordinates for the current patch
        patch_coords = (x, y, x + patch_size, y + patch_size)

        # Append the patch coordinates to the list
        coordinates_list.append(patch_coords)

        # Crop the patch from the image and mask
        image_patch = image.crop(patch_coords)
        mask_patch = mask.crop(patch_coords)

        # Create subdirectories for each image and mask pair
        image_subdir = os.path.join(output_image_dir, image_number)
        mask_subdir = os.path.join(output_mask_dir, image_number)
        
        os.makedirs(image_subdir, exist_ok=True)
        os.makedirs(mask_subdir, exist_ok=True)

        # Save the patch in the corresponding subdirectories
        image_patch.save(os.path.join(image_subdir, f"image_patch_{index}.png"), format="PNG")
        mask_patch.save(os.path.join(mask_subdir, f"mask_patch_{index}.png"), format="PNG")

        index += 1


'''
mask_directory = "/Users/krishsarin/Downloads/Krish/Testing/mask/120342"

# Create a list to store the coordinates of cancer patches
cancer_patch_coordinates = []

# Iterate through the files in the mask directory
for mask_filename in os.listdir(mask_directory):
    if mask_filename.endswith(".png"):
        # Extract the patch index from the mask filename
        patch_index = int(mask_filename.split("_")[-1].split(".")[0])

        # Use the patch index to get the corresponding coordinates from the coordinates_list
        if 0 <= patch_index < len(coordinates_list):
            patch_coordinates = coordinates_list[patch_index]

            # Append the patch coordinates to the cancer_patch_coordinates list
            cancer_patch_coordinates.append(patch_coordinates)
'''



'''
# Define the dimensions of the level one image
wsi_width_level1 = 34036
wsi_height_level1 = 15051

# Create an empty full mask with the correct dimensions
full_mask = np.zeros((wsi_height_level1, wsi_width_level1), dtype=np.uint8)

# Now, you can iterate through patch positions and add each mask to the corresponding position in the full_mask
for (y, x), mask in zip(patch_positions, all_masks):
    # Calculate the height and width of the mask
    mask_height, mask_width = mask.shape
    
    # Check if the mask goes out of bounds
    if y + mask_height > wsi_height_level1 or x + mask_width > wsi_width_level1:
        print(f"Mask at position ({x}, {y}) is out of bounds.")
    else:
        # Add the mask to the corresponding position in the full_mask
        full_mask[y:y+mask_height, x:x+mask_width] += mask

plt.imshow(full_mask, cmap='gray')
plt.axis('off')  # Turn off axis labels and ticks
plt.show()




print(f"Number of masks: {len(all_masks)}")
print(f"Number of patch positions: {len(patch_positions)}")

mask_dimensions = set(mask.shape[:2] for mask in all_masks)
if len(mask_dimensions) > 1:
    print("Error: Masks have different dimensions.")
else:
    print(f"Mask dimensions: {mask_dimensions.pop()}")

# Debugging: Print the sum of all masks to check if they contain non-zero values
print(f"Sum of all masks: {sum(mask.sum() for mask in all_masks)}")

cancer_mask = np.zeros_like(full_mask, dtype=np.uint8)

# Iterate through the cancerous_patch_indices and insert each cancerous patch into the cancer_mask
for index in cancerous_patch_indices:
    y, x = patch_positions[index]  # Get the position of the patch
    cancer_patch = all_masks[index]  # Get the cancerous patch

    # Insert the cancerous patch into the cancer_mask
    cancer_mask[y:y+patch_height, x:x+patch_width] = cancer_patch

plt.imshow(cancer_mask)

# Visualize the full mask
plt.imshow(cancer_mask, cmap='gray')

# Overlay patch positions
for y, x in patch_positions:
    plt.plot(x, y, 'ro')  # 'ro' means red circle marker
plt.show()



print("Cancer Patch Coordinates:")
for coord in cancer_patch_coordinates:
    print(coord)

# Print the contents of patch_positions
print("Patch Positions:")
for coord in patch_positions:
    print(coord)
'''

'''



# Assuming you have an empty mask created previously
empty_mask = np.zeros((wsi_height_level1, wsi_width_level1), dtype=np.uint8)

# Initialize a counter for the cancerous patches
cancer_patch_count = 0

# Iterate through all masks and patch positions
for mask, (y, x) in zip(all_masks, patch_positions):
    # Check if the current mask corresponds to a cancerous patch
    if cancer_patch_count < len(cancer_patch_coordinates) and (y, x) == cancer_patch_coordinates[cancer_patch_count]:
        # Apply the cancer mask to the empty mask at the correct coordinates
        empty_mask[y:y+patch_height, x:x+patch_width] = mask
        cancer_patch_count += 1
        print(cancer_patch_count)
print(cancer_patch_count)











predictions = model.predict(input_data)

predicted_mask = predictions[0]


smoothed_mask = cv2.GaussianBlur(predicted_mask, (0, 0), sigmaX=3)


threshold = 0.5  
binary_mask = (smoothed_mask > threshold).astype(np.uint8)


# Display the input image and the post-processed mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap='gray')
plt.title("Post-Processed Mask")
plt.axis("off")

plt.show()
'''


