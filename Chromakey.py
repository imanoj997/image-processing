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

def rgb_to_hsb(img):
    """
    This function converts RGB image to HSB color spaces
    and displays it in a collage of original image alongside
    3 grayscale versions of each band of HSB color spaces.

    Args:
    - img (cv2:img): image object of cv2

    Returns:
    - None
    """
    # Display the image`ß
    cv.imshow("Original Image Display", img)

    hsb_image = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    # cv.imshow("hsb",hsb_image)

    # split the hsb image into individual channels
    h,s,b = cv.split(hsb_image)

    # Stack images horizontally
    top_row = np.hstack([img, b])
    bottom_row = np.hstack([h, s])

    # Stack images vertically
    final_img = np.vstack([top_row, bottom_row])

    # Display the final image
    cv.imshow('Four Images', final_img)

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
        # Read the image
        image = cv.imread(image_path)

        # Check if image was successfully loaded
        if image is None:
            print("Error loading image")
            exit()
        else:
            rgb_to_hsb(image)
            pass

    # code for task 2
    else:
        #runtask2
        print("task 2-------------")
        return

if __name__ == "__main__":
    # start python program execution
    main()

