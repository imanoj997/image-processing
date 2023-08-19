# importing necessary python packages
import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def resize_to_larger(img1, img2):
    """
    This functions resizes the images to make them same size
    and aspect ratio. Bigger size of the two is choosen.
    Also ensure the max size in not more than 1280*720

    Args:
    - img1 (cv2:img): first image to be resized
    - img2 (cv2:img): second image to be resized

    Returns:
    - Resized images (cv2:img)
    """
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Find the target dimensions
    target_width = max(w1, w2)
    target_height = max(h1, h2)

    # Aspect ratio of the two images
    aspect1 = w1 / h1
    aspect2 = w2 / h2

    # Target aspect ratio: Use the larger one to avoid shrinking any image
    target_aspect = max(aspect1, aspect2)

    # Calculate dimensions with respect to aspect ratio
    if target_aspect > 1:  # width > height
        target_width = int(target_height * target_aspect)
    else:  # height >= width
        target_height = int(target_width / target_aspect)

    # Ensure that the target dimensions don't exceed 1280x720
    if target_width > 1280 or target_height > 720:
        # Resize dimensions proportionally to fit within the 1280x720 limit
        if target_width / target_height > 1280 / 720:  # Limit by width
            target_width = 1280
            target_height = int(target_width / target_aspect)
        else:  # Limit by height
            target_height = 720
            target_width = int(target_height * target_aspect)

    # Resize the images
    resized_img1 = cv.resize(img1, (target_width, target_height), interpolation=cv.INTER_LINEAR)
    resized_img2 = cv.resize(img2, (target_width, target_height), interpolation=cv.INTER_LINEAR)

    return resized_img1, resized_img2


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
    # Check the final image's dimensions to ensure it is within 1280*720
    height, width = image.shape[:2]

    # Calculate scaling factors
    scale_width = 1280 / width
    scale_height = 720 / height
    scale = min(scale_width, scale_height)
    # If any scaling factor is less than 1, resize the image maintaining aspect ratio
    if scale < 1:
        image = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

    # display the final image
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


def chorma_keying(green_screen_img, background_img):
    """
    This function converts performs chroma keying techniques.
    Replaces green background screen from and image and replaces it
    with another background image.

    Args:
    - green_screen_img (cv2:img): cv2 image object of subject with greenscreen background
    - background_img (cv2:img): cv2 image object of background to put subject in
    """
    # Convert greenscreen image to HSB for better object separation
    green_screen_img_hsb = cv.cvtColor(green_screen_img, cv.COLOR_BGR2HSV)

    # Set thresholds for green color in terms of HSV chanels
    lower_green_thresholds = np.array([35, 60, 40])
    upper_green_thresholds = np.array([90, 255, 255])

    # Set a mask to isolate subject from green background
    mask = cv.inRange(green_screen_img_hsb, lower_green_thresholds, upper_green_thresholds)

    # Refine the mask
    kernel = np.ones((5,5),np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Extract the subject
    subject = cv.bitwise_and(green_screen_img, green_screen_img, mask=~mask)

    # Extracted subject with white background
    extracted_subject = subject + cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    # Place subject on scenic background
    scenic_opening = cv.bitwise_and(background_img, background_img, mask=mask)
    composite = background_img + subject

    # Stack images horizontally
    top_row = np.hstack([green_screen_img, extracted_subject])
    bottom_row = np.hstack([background_img, composite])

    # Stack images vertically
    final_img = np.vstack([top_row, bottom_row])

    # Display the final collage image
    cv.imshow('Chroma Keying Results', final_img)
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
            print("Error: Cannot load image, might be a corrupted image.")
            exit()
        else:
            # calling crossponding color space conversion function
            rgb_to_color_spaces(image,color_space.lower())

    # code for task 2
    else:
        # Extract the image path
        background_img_path = sys.argv[1]
        green_screen_path = sys.argv[2]

        # check if paths are valid
        check_image_exists(background_img_path)
        check_image_exists(green_screen_path)

        # Read the image
        green_screen_img = cv.imread(green_screen_path)
        background_img = cv.imread(background_img_path)

        # Check if image was successfully loaded
        if green_screen_img is None or background_img is None:
            print("Error: Cannot load image, might be a corrupted image.")
            exit()
        else:
            # Convert to HSB color space for better color segmentation
            green_screen_img, background_img = resize_to_larger(green_screen_img, background_img)
            chorma_keying(green_screen_img, background_img)


if __name__ == "__main__":
    # start python program execution
    main()

