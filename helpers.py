# Helper functions

import os
import glob # library for loading images from a directory

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["red", "yellow", "green"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list

# This function converts an image to HSV colorspace
# and visualizes the individual color channels7
def visualize_hsv(rgb_image):
    
    # Convert to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # HSV channels
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    # Plot the original image and the three channels
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
    ax1.set_title('RGB image')
    ax1.imshow(rgb_image)
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')
    
def differentiate(vector):
    
    derivative = []
    
    prev_y = vector[0]
    
    dx = 1 #1 row or column of pixels
    
    for i in range(1, len(vector)):
        y = vector[i]
        dy = y - prev_y
        dy_by_dx = dy / dx
        derivative.append(dy_by_dx)
        
        prev_y = y

    return derivative

def crop(rgb_image): #optimized for the brightness classifier
    
    cropped_image = rgb_image[6:-6, 9:-9, :] #values chosen by trial and error.
    
    return cropped_image

def crop2(rgb_image): #optimized for crop_individual_section() function
    
    cropped_image = rgb_image[4:-5, 7:-7, :] #values chosen by trial and error.
    
    return cropped_image

def crop_individual_section(rgb_image, label):
   
    cropped = crop2(rgb_image)
    height = cropped.shape[0]
    red_section_start = 0
    yellow_section_start = int(np.ceil(height / 3))
    green_section_start = int(np.ceil(2 * height / 3))
    
    if label == [1, 0, 0]:
        selected_section = cropped[red_section_start:yellow_section_start, :, :]
    elif label == [0, 1, 0]:
        selected_section = cropped[yellow_section_start:green_section_start, :, :]
    elif label == [0, 0, 1]:
        selected_section = cropped[green_section_start:, :, :]
    else:
        return void
    
    return selected_section
    
def mask_brightness(rgb_image):
    
    lower_range = np.array([0, 0, 0])
    upper_range = np.array([100, 100, 100]) #values chosen by trial and error.
    
    mask = cv2.inRange(rgb_image, lower_range, upper_range)
    masked_image = np.copy(rgb_image)
    masked_image[mask != 0] = [0, 0, 0]
    
    return masked_image

def mask_saturation(rgb_image):
    
    hsv_image = cv2.cvtColor((rgb_image), cv2.COLOR_RGB2HSV)
    
    s = hsv_image[:,:,1]
    
    mask = cv2.inRange(s, 0, 75) #values chosen by trial and error.
    masked_image = np.copy(rgb_image)
    masked_image[mask != 0] = [0, 0, 0]
    
    return masked_image

