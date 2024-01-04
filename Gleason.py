#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:35:05 2023

@author: krishsarin
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute
from tensorflow.keras.utils import to_categorical
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.callbacks import Callback
from keras.models import Model
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import cv2
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from PIL import UnidentifiedImageError
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
import json


def stain_norm(input_image):
    # Convert the input image to a format expected by the function
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    Io = 240
    alpha = 1
    beta = 0.15
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape((-1, 3))
    OD = -np.log10((img.astype(float) + 1) / Io)

    ODhat = OD[~np.any(OD < beta, axis=1)]

    # Regularize the covariance matrix
    epsilon = 1e-5
    cov_matrix = np.cov(ODhat.T)
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon  # Add a small constant epsilon to the diagonal

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Inorm


#Preprocessing functions

def apply_stain_norm(image_path):
    image = cv2.imread(image_path)
    normalized_image = stain_norm(image)
    return normalized_image

def load_and_preprocess_data(directory, stain_norm_function, target_size=(512, 512), file_extension=".png"):
    filenames = []
    images = []
    labels = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(file_extension):
                file_path = os.path.join(root, filename)
                
                try:
                    
                    img = load_img(file_path, target_size=target_size)
                    img_array = img_to_array(img)
                    img_normalized = stain_norm_function(file_path)
                    
                    class_label = os.path.basename(root)
                    filenames.append(filename)
                    images.append(img_normalized)
                    labels.append(int(class_label))  
                except UnidentifiedImageError as e:
                    print(f"Error loading image: {file_path}. {e}")

    return np.array(images), np.array(labels), filenames

train_data_dir = "/Users/krishsarin/Downloads/Krish/new/Gleason/Training"
val_data_dir = "/Users/krishsarin/Downloads/Krish/new/Gleason/Validation"

#One hot encoding and adding dims to make data match model output

X_train, y_train, train_filenames = load_and_preprocess_data(train_data_dir, apply_stain_norm)


X_val, y_val, val_filenames = load_and_preprocess_data(val_data_dir, apply_stain_norm)

class_mapping = {0: 0, 3: 1, 4: 2}


y_train_mapped = np.vectorize(class_mapping.get)(y_train)


y_train_categorical = to_categorical(y_train_mapped, 3)

y_train_categorical_reshaped = np.expand_dims(y_train_categorical, axis=1)
y_train_categorical_reshaped = np.expand_dims(y_train_categorical_reshaped, axis=1)
y_train_categorical_reshaped = np.repeat(y_train_categorical_reshaped, 512, axis=1)
y_train_categorical_reshaped = np.repeat(y_train_categorical_reshaped, 512, axis=2)


print(y_train_categorical_reshaped.shape)


class_mapping = {0: 0, 3: 1, 4: 2}


y_val_mapped = np.vectorize(class_mapping.get)(y_val)


y_val_categorical = to_categorical(y_val_mapped, 3)

y_val_categorical_reshaped = np.expand_dims(y_val_categorical, axis=1)
y_val_categorical_reshaped = np.expand_dims(y_val_categorical_reshaped, axis=1)
y_val_categorical_reshaped = np.repeat(y_val_categorical_reshaped, 512, axis=1)
y_val_categorical_reshaped = np.repeat(y_val_categorical_reshaped, 512, axis=2)

print(y_val_categorical_reshaped.shape)

print("X_train shape:", X_val.shape)
print("y_train_categorical shape:", y_val_categorical.shape)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def custom_generator(X, y, batch_size):
    num_samples = len(X)
    while True:
        indices = np.random.choice(num_samples, batch_size, replace=False)
        batch_x = X[indices]
        batch_y = y[indices]
        yield batch_x, batch_y


batch_size = 8

train_generator = custom_generator(X_train, y_train_categorical, batch_size)
val_generator = custom_generator(X_val, y_val_categorical, batch_size)

#Visualizing batch

'''
def visualize_batch(generator):
    batch_x, batch_y = next(generator)
    
    for i in range(len(batch_x)):
        image = batch_x[i]
        label = batch_y[i]

        plt.imshow(image)
        plt.title(f'Label: {label}')
        plt.show()


visualize_batch(train_generator)
'''

# Directories
train_data_dir = "/Users/krishsarin/Downloads/Krish/new/Gleason/Training"
val_data_dir = "/Users/krishsarin/Downloads/Krish/new/Gleason/Validation"

#Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print(len(train_generator))
print(train_generator)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

datagen.fit(X_val)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(512, 512),
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)

#Data generator creation
custom_train_generator = custom_generator(train_generator, apply_stain_norm)


custom_batch_size = 8
num_batches = len(train_generator) // custom_batch_size


    
val_datagen = ImageDataGenerator(rescale=1./255)


val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(512, 512),
    batch_size=8,
    class_mode='categorical',
    shuffle=False  
)

#Val data generator creation
custom_val_generator = custom_generator(val_generator, apply_stain_norm)



#Class weight definitions

class_weights = {
    0: 1.0,
    1: 2.5,
    2: 3.0,
    }

total_samples = 322 + 137 + 289

weight_0 = total_samples / (3 * 322)
weight_1 = total_samples / (3 * 137)
weight_2 = total_samples / (3 * 289)

class_weighst = {
    0: weight_0,
    1: weight_1,
    2: weight_2,
}

print("Class Weights:", class_weights)

#Mobile Net V2 pre trained model
'''
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

for layer in base_model.layers:
    layer.trainable = False


x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)  # Adjusted dropout rate
x = Dense(64, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)


for layer in base_model.layers[-3:]:
    layer.trainable = True

model1 = Model(inputs=base_model.input, outputs=predictions)


model1.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


model1.summary()

lr_scheduler = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


model1.fit(datagen.flow(X_train, y_train_categorical, batch_size=batch_size),
           epochs=50,
           validation_data=(X_val, y_val_categorical),
           callbacks=[lr_scheduler, early_stopping])
'''

def lr_schedule(epoch):
    lr = 0.025
    if epoch > 25:
        lr *= 0.1
    return lr

#Efficient Net pre trained model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(512, 512, 3))


for layer in base_model.layers:
    layer.trainable = False


x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)  
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(3, activation='softmax')(x)  

model3 = Model(inputs=base_model.input, outputs=predictions)

model3.compile(optimizer=Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model3.summary()

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(512, 512, 3))


for layer in base_model.layers:
    layer.trainable = False


x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)  
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(2, activation='softmax')(x)  

model2 = Model(inputs=base_model.input, outputs=predictions)

model2.compile(optimizer=Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model2.summary()


loss, accuracy = model3.evaluate(X_val, y_val_categorical)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

lr_scheduler = LearningRateScheduler(lr_schedule)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)






model3.fit(datagen.flow(X_train, y_train_categorical, batch_size=batch_size),
           epochs=50,
           validation_data=(X_val, y_val_categorical),
           class_weight = class_weights,
           callbacks=[lr_scheduler, early_stopping])

model3.fit(
    X_train,
    y_train_categorical,
    #steps_per_epoch=len(X_train) // batch_size,
    epochs=50,
    validation_data=(X_val, y_val_categorical),
    #validation_steps=len(X_val) // batch_size,
    class_weight=class_weights,  
    callbacks=[early_stopping]  
)

model3.fit_generator(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=150,
    validation_data=val_generator,
    validation_steps=len(X_val) // batch_size,
    callbacks=[lr_schedule, early_stopping],
    class_weight = class_weights
)


model3.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[lr_scheduler]
)

model3.fit_generator(
    custom_train_generator,
    steps_per_epoch=len(custom_train_generator),
    epochs=75,
    validation_data=custom_val_generator,
    validation_steps=len(val_generator),
    callbacks = [early_stopping]
)

model3.summary()
# Saving weights
'''
model.save_weights('/Users/krishsarin/Downloads/Krish/Results/model/model3.h5')


class_weights_list = [class_weights[i] for i in range(len(class_weights))]


with open('/Users/krishsarin/Downloads/Krish/Results/model/class_weights.json', 'w') as json_file:
    json.dump(class_weights_list, json_file)
    
model3 = load_model("/Users/krishsarin/Downloads/Krish/Results/model/modelmid.h5")


with open('class_weights.json', 'r') as json_file:
    class_weights_list = json.load(json_file)


class_weights = {i: class_weights_list[i] for i in range(len(class_weights_list))}
'''

#Predictions for testing
directory_path = '/Users/krishsarin/Downloads/Krish/new/Gleason/Validation/Validation0'
target_size = (512, 512)


for filename in os.listdir(directory_path):
    if filename.endswith('.png'):
        
        img_path = os.path.join(directory_path, filename)

        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model3.predict(img_array)

        
        predicted_class = np.argmax(predictions[0])

        print(f"File: {filename}, Predicted class index: {predicted_class}")



def create_full_mask_from_biopsy1(biopsy_path, model, model2, model4):
    def simulate_save_and_load(image_array):
        image = Image.fromarray(image_array)
        return np.array(image)

    biopsy_image = Image.open(biopsy_path)
    biopsy_width, biopsy_height = biopsy_image.size

    patch_size = 512
    coordinates_list = []
    all_patches = []
    gleason_predictions = []

    for y in range(0, biopsy_height, patch_size):
        for x in range(0, biopsy_width, patch_size):
            patch_coords = (x, y, x + patch_size, y + patch_size)
            patch = biopsy_image.crop(patch_coords)
            image_patch_array = simulate_save_and_load(np.array(patch))

            try:
                img_array = stain_norm(image_patch_array)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
    
                predictions = model4.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                if predicted_class == 0:
                    gleason_predictions.append(0)
                elif predicted_class == 1:
                    predictions = new_model.predict(img_array)
                    predicted_class = np.argmax(predictions[0])
                    gleason_predictions.append(predicted_class + 1)
                
                all_patches.append(patch)
                coordinates_list.append(patch_coords)

            except Exception as e:
                print(f"Stain normalization or prediction failed for patch at ({x}, {y}): {e}")

                continue
    

    
    batch_size = 32  
    batches = [all_patches[i:i+batch_size] for i in range(0, len(all_patches), batch_size)]

    all_masks = []

    for batch, gleason_batch in zip(batches, [gleason_predictions[i:i+batch_size] for i in range(0, len(gleason_predictions), batch_size)]):
        batch = np.array(batch) / 255.0
        batch_predictions = model.predict(batch)

        for mask, gleason_score in zip(batch_predictions, gleason_batch):
            binary_mask = (mask[:, :, 0] > 0.5).astype(np.uint8)
            color_mask = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)

            if gleason_score == 1:
                color_mask[binary_mask == 1] = [255,255,0] # Yellow
            elif gleason_score == 2:
                color_mask[binary_mask == 1] = [255, 165, 0]  # Orange
            elif gleason_score == 3:
                color_mask[binary_mask == 1] = [255, 0, 0]    # Red
            

            all_masks.append(color_mask)

    full_mask = np.zeros((biopsy_height, biopsy_width, 3), dtype=np.uint8)

    for mask, (left, upper, right, lower) in zip(all_masks, coordinates_list):
        if mask is None:
            print(f"Skipping None mask at coordinates ({left}, {upper}, {right}, {lower})")
            continue

        adjusted_mask = mask.astype(np.uint8)

        try:
            full_mask[upper:lower, left:right] = np.maximum(full_mask[upper:lower, left:right], adjusted_mask)
        except ValueError as e:
            error_message = str(e)
            match = re.search(r'(\d+),(\d+)', error_message)
            if match:
                expected_height, expected_width = map(int, match.groups())
            else:
                print(f"Failed to extract expected dimensions from the error message.")
                continue

            adjusted_upper = upper
            adjusted_lower = lower + (512 - expected_height)
            adjusted_left = left
            adjusted_right = right + (512 - expected_width)

            scale_x = expected_width / mask.shape[1]
            scale_y = expected_height / mask.shape[0]

            if scale_x < 1 and scale_y < 1:
                adjusted_mask = cv2.resize(adjusted_mask, (0, 0), fx=scale_x, fy=scale_y)
                full_mask[adjusted_upper:adjusted_lower, adjusted_left:adjusted_right] = np.maximum(full_mask[adjusted_upper:adjusted_lower, adjusted_left:adjusted_right], adjusted_mask)

    full_mask_pil = Image.fromarray(full_mask)
    
    plt.imshow(full_mask_pil)
    plt.show()
    yellow = [255, 255, 0]
    orange = [255, 165, 0]
    red = [255, 0, 0]
    
    # Count pixels for each color
    yellow_count, orange_count, red_count = 0, 0, 0
    for mask in all_masks:
        yellow_count += np.sum(np.all(mask == yellow, axis=-1))
        orange_count += np.sum(np.all(mask == orange, axis=-1))
        red_count += np.sum(np.all(mask == red, axis=-1))
    
    total_colored = yellow_count + orange_count + red_count
    
    # Calculate percentages
    percentage_yellow = yellow_count / total_colored if total_colored else 0
    percentage_orange = orange_count / total_colored if total_colored else 0
    percentage_red = red_count / total_colored if total_colored else 0
    
    # Print percentages
    print(percentage_yellow, percentage_orange, percentage_red)
    
    majority_color = max((percentage_yellow, '3'), (percentage_orange, '4'), (percentage_red, '5'), key=lambda x: x[0])[1]

    if majority_color == '3':
        if percentage_orange <= 0.15 and percentage_red <= 0.15:
            print("3+3")
        elif percentage_orange > percentage_red:
            print("3+4")
        else:
            print("3+5")
    
    elif majority_color == '4':
        if percentage_yellow <= 0.15 and percentage_red <= 0.15:
            print("4+4")
        elif percentage_yellow > percentage_red:
            print("4+3")
        else:
            print("4+5")
    
    else:  
        if percentage_yellow <= 0.15 and percentage_orange <= 0.15:
            print("5+5")
        elif percentage_yellow > percentage_orange:
            print("5+3")
        else:
            print("5+4")


model = load_model('/Users/krishsarin/Downloads/Krish/Results/model/model.h5')
biopsy = '/Users/krishsarin/Downloads/Krish/level1/Gleason/120373_level_1_image.png'
create_full_mask_from_biopsy1(biopsy, model, new_model, model3)    


