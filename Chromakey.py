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
    - image_path (str): full path of the image to be checked

    Returns:
    - True of image exists, ends the program with a message if image doesn't exists.
    """
    if os.path.exists(image_path):
        return True
    else:
        print(f"Error: {image_path} does not exist.")
        exit()

def display_image(image_path):
    # Read the image
    image = cv.imread(image_path)

    # Check if image was successfully loaded
    if image is None:
        print("Error loading image")
        return

    # Display the image`ß
    cv.imshow('Image Display', image)

    # Wait for a key press and then close the image window
    cv.waitKey(0)
    cv.destroyAllWindows()

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
        # rgb_to_hsb()
    # code for task 2
    else:
        #runtask2
        print("task 2-------------")
        return

    # Display the image
    display_image(image_path)

if __name__ == '__main__':
    # start python program execution
    main()

