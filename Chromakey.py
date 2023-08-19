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
    # convert to hsb
    hsb_image = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)

    # split the hsb image into individual channels
    h,s,b = cv.split(hsb_image)

   # Convert single channel grayscale images to 3-channel grayscale images to display same window as 3-dimension original image
    def to_3channel_gray(single_channel):
        return cv.merge([single_channel, single_channel, single_channel])

    h_3channel = to_3channel_gray(h)
    s_3channel = to_3channel_gray(s)
    b_3channel = to_3channel_gray(b)

    # Stack images horizontally
    top_row = np.hstack([img, b_3channel])
    bottom_row = np.hstack([h_3channel, s_3channel])

    # Stack images vertically
    final_img = np.vstack([top_row, bottom_row])

    # Display the final image
    cv.imshow('Four Images', final_img)

    # Wait for a key press and then close the image window
    cv.waitKey(0)
    cv.destroyAllWindows()


def split_rgb(img):
    # split the hsb image into individual channels
    r,g,b = cv.split(img)

   # Convert 1-channel grayscale images to 3-channel grayscale images to display on same window as 3-dimension original image
    def to_3channel_gray(single_channel):
        return cv.merge([single_channel, single_channel, single_channel])

    r_3channel = to_3channel_gray(r)
    g_3channel = to_3channel_gray(g)
    b_3channel = to_3channel_gray(b)

    print(g_3channel[100,100])

    # Stack images horizontally
    top_row = np.hstack([img, b_3channel])
    bottom_row = np.hstack([r_3channel, g_3channel])

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
            # split_rgb(image)

    # code for task 2
    else:
        #runtask2
        print("task 2-------------")
        return

if __name__ == "__main__":
    # start python program execution
    main()

