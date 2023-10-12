#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:05:55 2023

@author: krishsarin
"""

from openslide import OpenSlide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os


svs_path = "/Users/krishsarin/Downloads/Krish/resized_masks/120375.svs"


slide = OpenSlide(svs_path)
dims = slide.level_dimensions
num_levels = len(dims)
level_dim = dims[2]
num_level=2
    #Give pixel coordinates (top left pixel in the original large image)


level_2_image = slide.read_region((0, 0), 2, slide.level_dimensions[2]).convert("RGB")
level_2_image.save("/Users/krishsarin/Downloads/Krish/120375_level_2_image.png", format="PNG")



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

index = 0

# Output directory for saving patches
image_output_dir = "/Users/krishsarin/Downloads/Krish/Validation/img1/120375" 
mask_output_dir = "/Users/krishsarin/Downloads/Krish/Validation/mask1/120375"

# Define the patch size
patch_size = 512

# Loop through the image to create patches
for y in range(0, image_height, patch_size):
    for x in range(0, image_width, patch_size):
        # Define the coordinates for the current patch
        patch_coords = (x, y, x + patch_size, y + patch_size)

        # Crop the patch from the image and mask
        image_patch = image.crop(patch_coords)
        mask_patch = mask_resize.crop(patch_coords)
        
        image_patch.save(os.path.join(image_output_dir, f"image_patch_{index}.png"), format="PNG")
        mask_patch.save(os.path.join(mask_output_dir, f"mask_patch_{index}.png"), format="PNG")
        
        index += 1
        
        plt.subplot(1, 2, 1)
        plt.imshow(image_patch)
        plt.title("Image Patch")

        plt.subplot(1, 2, 2)
        plt.imshow(mask_patch, cmap="gray")
        plt.title("Mask Patch")

        plt.show()
        


#slide.close()


#print(f"Level 1 Dimensions: Width = {level_1_dimensions[0]}, Height = {level_1_dimensions[1]}")
Image.MAX_IMAGE_PIXELS = 10000000000
mask_image_path = "/Users/krishsarin/Downloads/Krish/resized_masks/120346_mask.png"
mask_image = Image.open(mask_image_path)
#plt.imshow(mask_image)
mask_resize = mask_image.resize((level_img_np.shape[1], level_img_np.shape[0]))
plt.imshow(mask_resize)

#mask_resize.save("/Users/krishsarin/Downloads/Krish/resized_masks/120335_mask.png")

width, height = mask_resize.size

print(width, height)

patch_mask = mask_resize.crop((1000, 1500, 1512, 2012))

plt.imshow(patch_mask)
mask_image.close()

plt.figure(figsize=(8, 4))

# Display the slide patch
plt.subplot(1, 2, 1)
plt.imshow(patch_slide)
plt.title("Slide Patch 0")
plt.axis('off')

# Display the mask patch
plt.subplot(1, 2, 2)
plt.imshow(patch_mask, cmap='gray')  # Assuming the mask is grayscale
plt.title("Mask Patch 0")
plt.axis('off')

plt.tight_layout()
plt.show()



