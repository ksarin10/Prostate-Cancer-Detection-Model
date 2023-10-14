#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:05:55 2023

@author: krishsarin
"""
'''
The purpose of this program is to resize all of my wholeslide images into their level one size. The OpenSlide
library's level one images are slightly lower resolution than the wholeslide images but they are 
significantly easier to process. The original masks and images that I was given were not the same size, so
I used OpenSlide to resize the images to their level one dimensions, and I used the PIL library to resize
masks to the level one dimensions of the image. This is essential to the purpose as it allows me to break
down the images and masks into matching 512x512 patches that can successfully train my model, and save all 
these matching patches into directories. At the end of this  file, I have a code segment that checks all of 
the patches within the mask directories that I created and looks for all of the cancer mask patches within 
these directories and copies those masks and their into two respective directories that will be used for 
training, validation, and testing. This is essential as the model will only train properly if it is trained
with the cancer images. The blank masks(non-cancerous masks) significantly outnumber the cancer patches
and create a data imbalanace that stops the model from training properly.

'''


from openslide import OpenSlide
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import re
import cv2
from PIL import Image




input_path = "/Users/krishsarin/Downloads/Krish/level1/120342_level_1_image.png"



input_directory = "/Users/krishsarin/Downloads/Krish/resized_masks/"
output_directory = "/Users/krishsarin/Downloads/Krish/level1/"

# Get a list of all SVS files in the input directory
svs_files = [f for f in os.listdir(input_directory) if f.endswith(".svs")]

for svs_file in svs_files:
    svs_path = os.path.join(input_directory, svs_file)
    
    # Extract the image number from the file name 
    image_number = svs_file.split(".")[0]

    # Define the output path for the level 1 image
    output_path = os.path.join(output_directory, f"{image_number}_level_1_image.png")

    # Open the SVS slide
    slide = OpenSlide(svs_path)

    # Read the image at a higher resolution level and convert it to RGB
    higher_res_image = slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB")

    # Resize the higher resolution image to the level 1 dimensions using Lanczos resampling
    level_1_image = higher_res_image.resize(slide.level_dimensions[1], Image.LANCZOS)

    # Save the level 1 image as PNG
    level_1_image.save(output_path, format="PNG")

    print(f"Processed {svs_file} and saved as {output_path}")

svs_directory = "/Users/krishsarin/Downloads/Krish/level1/"
resized_mask_directory = "/Users/krishsarin/Downloads/Krish/level1/"

# Get a list of all level 1 SVS files in the directory
svs_files = [f for f in os.listdir(svs_directory) if f.endswith("_level_1_image.png")]

for svs_file in svs_files:
    Image.MAX_IMAGE_PIXELS = None
    # Extract the image number from the file name
    image_number = svs_file.split("_")[0]

    # Define the output path for the resized mask (in the level1 directory)
    resized_mask_path = os.path.join(resized_mask_directory, f"{image_number}_level_1_mask.png")

    # Open the level 1 SVS image to get its dimensions
    svs_path = os.path.join(svs_directory, svs_file)
    svs_image = Image.open(svs_path)
    
    # Get the dimensions of the level 1 SVS image
    level_1_dimensions = svs_image.size

    # Define the input path for the mask (assuming the mask file name format is "number_mask.png")
    mask_path = os.path.join('/Users/krishsarin/Downloads/Krish/resized_masks', f"{image_number}_mask.png")

    # Open the mask image
    mask_image = Image.open(mask_path)

    # Resize the mask to the level 1 dimensions using Lanczos resampling
    resized_mask = mask_image.resize(level_1_dimensions, Image.LANCZOS)

    # Save the resized mask as PNG in the level1 directory
    resized_mask.save(resized_mask_path, format="PNG")

    print(f"Processed {mask_path} and saved as {resized_mask_path}")

svs_directory = "/Users/krishsarin/Downloads/Krish/level1/"

# Get a list of all image files in the level1 directory
image_files = [f for f in os.listdir(svs_directory) if f.endswith(".png")]

for image_file in image_files:
    # Define the full path to the image
    image_path = os.path.join(svs_directory, image_file)

    # Open the image to get its dimensions
    image = Image.open(image_path)

    # Get and print the dimensions of the image
    dimensions = image.size
    print(f"Image: {image_file}, Dimensions: {dimensions}")


# Load the image and mask
image_path = "/Users/krishsarin/Downloads/Krish/120375_level_2_image.png"  # Replace with the actual path to your image
mask_path = "/Users/krishsarin/Downloads/Krish/original_masks/120375_mask.png"    # Replace with the actual path to your mask
image = Image.open(image_path)
image_width, image_height = image.size
mask = Image.open(mask_path)
mask_resize = mask.resize((image_width, image_height))
plt.imshow(mask_resize)
plt.imshow(image)
mask_width, mask_height = mask_resize.size
print(image_width, image_height)
print(mask_width, mask_height)
# Get the dimensions of the image



# Output directories for saving patches
output_image_dir = "/Users/krishsarin/Downloads/Krish/level1/output_img"
output_mask_dir = "/Users/krishsarin/Downloads/Krish/level1/output_mask"

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Patch size
patch_size = 512
level1_directory = "/Users/krishsarin/Downloads/Krish/level1"

# Get a list of image files in the "level1" directory
image_files = [f for f in os.listdir(level1_directory) if f.endswith("_image.png")]

for image_file in image_files:
    # Extract the image number from the image file name
    image_number = image_file.split("_")[0]
    
    # Construct the corresponding mask file name
    mask_file = f"{image_number}_level_1_mask.png"
    
    # Open the image and mask
    image_path = os.path.join(level1_directory, image_file)
    mask_path = os.path.join(level1_directory, mask_file)
    
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    # Get image dimensions
    image_width, image_height = image.size
    
    # Loop through the image to create patches
    index = 0  # Reset index for each image
    for y in range(0, image_height, patch_size):
        for x in range(0, image_width, patch_size):
            # Define the coordinates for the current patch
            patch_coords = (x, y, x + patch_size, y + patch_size)

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

            index += 1  # Increment index for the next patch






            
            
        '''
        plt.subplot(1, 2, 1)
        plt.imshow(image_patch)
        plt.title("Image Patch")

        plt.subplot(1, 2, 2)
        plt.imshow(mask_patch, cmap="gray")
        plt.title("Mask Patch")

        plt.show()
        '''




# Directory paths
svs_dir = "/Users/krishsarin/Downloads/Krish/resized_masks"
image_output_dir = "/Users/krishsarin/Downloads/Krish/Training/img2"
mask_output_dir = "/Users/krishsarin/Downloads/Krish/Training/mask2"

# Patch size
patch_size = 512

# Loop through SVS files in the directory
for filename in os.listdir(svs_dir):
    if filename.endswith(".svs"):
        # Get the SVS file path and corresponding mask path
        svs_path = os.path.join(svs_dir, filename)
        mask_path = os.path.join("/Users/krishsarin/Downloads/Krish/resized_masks", filename.replace(".svs", "_mask.png"))

        # Open the SVS file
        slide = OpenSlide(svs_path)
        dims = slide.level_dimensions

        # Get the dimensions of the level one image
        level_two_dim = dims[2]
        image_width, image_height = level_two_dim  # Assign image dimensions

        # Resize the SVS image to match the level one dimensions
        svs_image = slide.read_region((0, 0), 2, level_two_dim).convert("RGB")

        # Open the mask and resize it to match the image dimensions
        Image.MAX_IMAGE_PIXELS = None
        mask = Image.open(mask_path)
        mask = mask.resize(level_two_dim)

        image_filename = os.path.splitext(os.path.basename(svs_path))[0]  # Get the filename without extension
        image_subdir = os.path.join(image_output_dir, image_filename)
        mask_subdir = os.path.join(mask_output_dir, image_filename)

        os.makedirs(image_subdir, exist_ok=True)
        os.makedirs(mask_subdir, exist_ok=True)

        index = 0  # Initialize the index to 0

        # Loop through the image to create patches
        for y in range(0, image_height, patch_size):
            for x in range(0, image_width, patch_size):
                # Define the coordinates for the current patch
                patch_coords = (x, y, x + patch_size, y + patch_size)
        
                # Crop the patch from the image and mask
                image_patch = svs_image.crop(patch_coords)
                mask_patch = mask.crop(patch_coords)
        
                # Save the patches with sequential numbering
                image_patch.save(os.path.join(image_subdir, f"image_patch_{index}.png"), format="PNG")
                mask_patch.save(os.path.join(mask_subdir, f"mask_patch_{index}.png"), format="PNG")
        
                index += 1  # Increment the index for the next patch
                



train_image_dir = "/Users/krishsarin/Downloads/Krish/level1/output_img/120375"
train_mask_dir = "/Users/krishsarin/Downloads/Krish/level1/output_mask/120375"

# Output base directory
output_base_dir = "/Users/krishsarin/Downloads/Krish/Training/level1"

# Output directories for images and masks
output_image_dir = os.path.join(output_base_dir, "img2c")
output_mask_dir = os.path.join(output_base_dir, "mask2c")

# Create output directories for images and masks if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Copy the patch directory (e.g., "120378") into the img2c and mask2c directories
output_image_patch_dir = os.path.join(output_image_dir, os.path.basename(train_image_dir))
output_mask_patch_dir = os.path.join(output_mask_dir, os.path.basename(train_mask_dir))

# Create output directories for the patch directories
os.makedirs(output_image_patch_dir, exist_ok=True)
os.makedirs(output_mask_patch_dir, exist_ok=True)

# List image files in the image directory
image_files = [f for f in os.listdir(train_image_dir) if f.endswith(".png")]

# Traverse the list of image files
for image_file in image_files:
    image_path = os.path.join(train_image_dir, image_file)
    
    # Extract the patch number from the image file name
    image_patch_number = int(image_file.split('_')[-1].split('.')[0])
    
    # Create the corresponding mask file name
    mask_file = f"mask_patch_{image_patch_number}.png"
    mask_path = os.path.join(train_mask_dir, mask_file)

    # Check if the mask file exists
    if os.path.exists(mask_path):
        # Load the mask
        mask = Image.open(mask_path)

        # Analyze the mask 
        mask_array = np.array(mask)
        has_cancer = np.any(mask_array > 0)

        # If the mask has cancer annotations, copy the image and mask to the output directory
        if has_cancer:
            print(f"Processing image: {image_path}")
            print(f"Matching mask: {mask_path}")
            
            # Copy the image and mask to the patch directories
            image_output_path = os.path.join(output_image_patch_dir, image_file)
            mask_output_path = os.path.join(output_mask_patch_dir, mask_file)
            print(f"Copying image to: {image_output_path}")
            print(f"Copying mask to: {mask_output_path}")
            shutil.copy(image_path, image_output_path)
            shutil.copy(mask_path, mask_output_path)
        else:
            print(f"No cancer annotations found in mask: {mask_path}")
    else:
        print(f"Mask file does not exist for image: {image_path}")

print("Processing complete.")



