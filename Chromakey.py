# importing necessary python packages
import sys
import os
import cv2 as cv
import numpy as np
import matplotlib as plt

def check_image_exists(image_path):
    """
    This function check if an image with a given path exists or not

    Args:
    - image (str): full path of the image to be checked

    Returns:
    - True of image exists, ends the program with a message if image doesn't exists.
    """
    if os.path.exists(image_path):
        return True
    else:
        print(f"Error: {image_path} does not exist.")
        exit()

def display_image(image):
    """
    This function display an image using opencv's imshow function.

    Args:
    - image: image to be displayed
    """
    cv.imshow('Four Images', image) # Displaying image

    # Wait for a key press and then close the image window
    cv.waitKey(0)
    cv.destroyAllWindows()


def to_3channel_gray(single_channel_image):
    """
    This function changes single channel grayscale image to
    3 channel grayscale image.

    Args:
    - single_channel_image: single channel grayscale image to convert

    Returns:
    - 3 channel grayscale image
    """
    return cv.merge([single_channel_image, single_channel_image, single_channel_image])


def rgb_to_hsb(img):
    """
    This function converts RGB image to HSB color spaces
    and displays it in a collage of original image alongside
    3 grayscale versions of each channel of HSB color spaces.

    Args:
    - img (cv2:img): image object of cv2 in rgb
    """
    # convert to hsb
    hsb_image = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)

    # split the hsb image into individual channels
    h,s,b = cv.split(hsb_image)

   # Convert single channel grayscale images to 3-channel grayscale images because single channel grayscale
   # and 3-channel original rgb image cannot be displayed in same window
    h_3channel =  to_3channel_gray(h)
    s_3channel = to_3channel_gray(s)
    b_3channel = to_3channel_gray(b)

    # Stack images horizontally
    top_row = np.hstack([img, b_3channel])
    bottom_row = np.hstack([h_3channel, s_3channel])

    # Stack images vertically
    final_img = np.vstack([top_row, bottom_row])

    # Display the final image
    display_image(final_img)


def rgb_to_lab(img):
    """
    This function converts RGB image to Lab color spaces
    and displays it in a collage of original image alongside
    3 grayscale versions of each channel of Lab color spaces.

    Args:
    - img (cv2:img): image object of cv2 in rgb
    """
    # convert to hsb
    lab_image = cv.cvtColor(img, cv.COLOR_BGR2Lab)

    # split the hsb image into individual channels
    l_grayscale, a_grayscale, b_grayscale = cv.split(lab_image)

   # Convert single channel grayscale images to 3-channel grayscale images because single channel grayscale 
   # and 3-channel original rgb image cannot be displayed in same window
    l_3channel =  to_3channel_gray(l_grayscale)
    a_3channel = to_3channel_gray(a_grayscale)
    b_3channel = to_3channel_gray(b_grayscale)

    # normalizing the pixel values to range of 0-255
    a_3channel = cv.normalize(a_3channel, None, 0, 255, cv.NORM_MINMAX)
    b_3channel = cv.normalize(b_3channel, None, 0, 255, cv.NORM_MINMAX)

    # Stack images horizontally
    top_row = np.hstack([img, l_3channel])
    bottom_row = np.hstack([a_3channel, b_3channel])

    # Stack images vertically
    final_img = np.vstack([top_row, bottom_row])

    # Display the final image
    display_image(final_img)


def main():
    if len(sys.argv) != 3:
        print("Error: Invalid command \n Use 'python Chromakey.py -color_space image_path' to run task 1 \n \
        Use 'python Chromakey.py scenicImageFile greenScreenImagefile' to run task 2")
        return

    # code for task 1
    if not '.' in sys.argv[1]:
        # Extract the argument without the "-" to identify color space
        color_space = sys.argv[1][1:]
        if color_space not in ("XYZ", "Lab", "YCrCb", "HSB"):
            print("Error: Invalid color space. Valid options are 'XYZ', 'Lab', 'YCrCb', 'HSB'")
            return

        # Create a dictionary to map function names to function references
        function_mapping = {
                "hsb": rgb_to_hsb,
                # "xyz": rgb_to_xyz,
                "lab": rgb_to_lab,
                # "ycrcb": rgb_to_ycrcb
            }

        # Extract the image path
        image_path = sys.argv[2]
        check_image_exists(image_path)
        # Read the image
        image = cv.imread(image_path)

        # Check if image was successfully loaded
        if image is None:
            print("Error loading image")
            exit()
        else:
            # calling crossponding color space conversion function
            function_mapping[color_space.lower()](image)
            # split_rgb(image)

    # code for task 2
    else:
        #runtask2
        print("task 2-------------")
        return

if __name__ == "__main__":
    # start python program execution
    main()

