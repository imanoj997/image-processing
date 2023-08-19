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


def rgb_to_color_spaces(img, color_space):
    """
    This function converts RGB image to other color spaces
    and displays it in a collage of original image alongside
    3 grayscale versions of each channel of converted color spaces.

    Args:
    - img (cv2:img): cv2 image object of rgb image
    - color_space (str): color space to convert image into
    """
    # convert to hsb
    cv_converter_mapping = {
        "hsb": cv.COLOR_BGR2HSV_FULL,
        "lab": cv.COLOR_BGR2Lab,
        "xyz": cv.COLOR_BGR2XYZ,
        "ycrcb": cv.COLOR_BGR2YCrCb
    }
    converted_image = cv.cvtColor(img, cv_converter_mapping[color_space])

    # split the hsb image into individual channels
    first_color_grayscale, second_color_grayscale, third_color_grayscale = cv.split(converted_image)

   # Convert single channel grayscale images to 3-channel grayscale images because single channel grayscale 
   # and 3-channel original rgb image cannot be displayed in same window
    first_color_3grayscale =  to_3channel_gray(first_color_grayscale)
    second_color_3grayscale = to_3channel_gray(second_color_grayscale)
    third_color_3grayscale = to_3channel_gray(third_color_grayscale)

    if color_space in ("lab", "ycrcb"):
        # normalizing the a and b pixel values to range of 0-255 for Lab color space because they originally have [-127,127] range
        # normalizing the Cr and Cb pixel values to range of 0-255 for YCrCb color space because they originally have [16, 240] range
        second_color_3grayscale = cv.normalize(second_color_3grayscale, None, 0, 255, cv.NORM_MINMAX)
        third_color_3grayscale = cv.normalize(third_color_3grayscale, None, 0, 255, cv.NORM_MINMAX)
        if color_space == "ycrcb":
            # normalizing the Y pixel values to range of 0-255 for YCrCb color space because they originally have [16, 235] range
            first_color_3grayscale = cv.normalize(first_color_3grayscale, None, 0, 255, cv.NORM_MINMAX)

    # Stack images horizontally
    if color_space == "hsb":
        # for hsb color space 1st, 2nd and 3rd channels are to be displayed in 3rd, 4th and 1st quadrants respectively
        top_row = np.hstack([img, third_color_3grayscale])
        bottom_row = np.hstack([first_color_3grayscale, second_color_3grayscale])
    else:
        # for all other color spaces 1st, 2nd and 3rd channels are to be displayed in 1st, 3rd and 4th quadrants respectively
        top_row = np.hstack([img, first_color_3grayscale])
        bottom_row = np.hstack([second_color_3grayscale, third_color_3grayscale])

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
            rgb_to_color_spaces(image,color_space.lower())

    # code for task 2
    else:
        #runtask2
        print("task 2-------------")
        return

if __name__ == "__main__":
    # start python program execution
    main()

