# Prostate-Cancer-Detection-Model

The purpose of this resizing.py is to resize all of my wholeslide images into their level one size. The OpenSlide
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

This file takes the patches from the resizing.py file and trains the model using those patches. The model 
is created using a U-Net architecture which is optimal for image segmentation processes. Before I train the
model I read all of the images using cv2 and put them through stain normalization, which is a technique 
to make all of the biopsies the same color and have the model produce the optimal results. After training
the model, I created a few functions to create all of the mask predictions and determine the correct 
coordinates for all of the masks. Then I created an empty mask that is the same size as the real mask 
from my dataset and I inserted all  of the masks in to their correct coordinates to get the final 
cancer mask.

Gleason.py is a file that is able to classify a biopsy image based on it's gleason score, the medical standard
of how severe the prostate cancer is, using the EfficientNetB0 model architecture. Within this file, I've developed a
function combining the classification with the U-Net segmentation model in order to produce a colored biopsy mask that 
not only segments the cancer within the biopsy image but color codes it based on it's severity on the gleason scale.
