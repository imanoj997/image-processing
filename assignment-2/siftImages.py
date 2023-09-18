# importing necessary python packages
import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

def resize_to_vga(img):
    """
    This function resizes the image to be roughly VGA size (480x600) while maintaining its aspect ratio.

    Args:
    - img (cv2:img): image to be resized

    Returns:
    - Resized image (cv2:img)
    """
    # Get image dimensions
    h, w = img.shape[:2]

    # Target VGA dimensions
    target_width = 600
    target_height = 480

    # Image's aspect ratio
    aspect = w / h

    # VGA's aspect ratio
    vga_aspect = target_width / target_height

    # Adjust target dimensions based on the aspect ratio of the image
    if aspect > vga_aspect:
        # Width is the limiting factor, adjust height accordingly
        target_height = int(target_width / aspect)
    else:
        # Height is the limiting factor, adjust width accordingly
        target_width = int(target_height * aspect)

    # Resize the image
    resized_img = cv.resize(img, (target_width, target_height), interpolation=cv.INTER_LINEAR)

    return resized_img



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


def display_image(image, window_name="Image"):
    """
    This function check the input image's dimensions to ensure it is less than 1280*720,
    then displays the image using opencv's imshow function.

    Args:
    - image (cv2: img): image to be displayed
    - window_name (str): name of the window to display image
    """
    height, width = image.shape[:2]

    # Calculate scaling factors
    scale_width = 1280 / width
    scale_height = 720 / height
    scale = min(scale_width, scale_height)
    # If any scaling factor is less than 1, resize the image maintaining aspect ratio
    if scale < 1:
        image = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

    # Display the final image
    cv.imshow(window_name, image) # Displaying image

    # Wait for a key press and then close the image window
    cv.waitKey(0)
    cv.destroyAllWindows()


def sift_extractor(image, image_name):
    """
    Extracts SIFT keypoints from the luminance (Y) component of an image 
    and displays the image with the keypoints highlighted.

    Args:
    - image (cv2: img): Image from which to extract SIFT keypoints.
    - image_name (str): name of the image
    """
    work_image = image.copy()
    yuv = cv.cvtColor(work_image, cv.COLOR_BGR2YUV)
    # Initialize the SIFT detector
    sift = cv.SIFT_create()
    # Extract the Y (luminance) component from the YUV image
    y_channel = yuv[:,:,0]

    # Detect SIFT keypoints from the Y component
    kp = sift.detect(y_channel, None)

    # Loop through each detected keypoint to draw a cross
    for keypoint in kp:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        line_length = 5
        cv.line(work_image, (x - line_length, y), (x + line_length, y), (0, 0, 255), 1)
        cv.line(work_image, (x, y - line_length), (x, y + line_length), (0, 0, 255), 1)

    # Draw the keypoints with circles and orientation lines on the working image
    image_w_keypoints = cv.drawKeypoints(y_channel, kp, work_image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Stack the original image and the image with keypoints side-by-side for comparison
    final_image = np.hstack([image, image_w_keypoints])
    display_image(final_image)
    print(f"# of keypoints in {image_name} is {len(kp)}")


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

def main():
    if len(sys.argv) == 1:
        print("Error: Invalid command \n Use 'python siftImages.py image-path' to run task 1 \n \
        Use 'python siftImages.py image-path1 image-path2.....' to run task 2")
        return

    # code for task 1
    if len(sys.argv) == 2:
        # Extract the image path
        image_path = sys.argv[1]
        check_image_exists(image_path)
        # Read the image
        image = cv.imread(image_path)
        image_name = os.path.basename(image_path)

        # Check if image was successfully loaded
        if image is None:
            print("Error: Cannot load image, might be a corrupted image.")
            exit()
        else:
            resized_image = resize_to_vga(image) # resize the image close to VGA size
            sift_extractor(resized_image, image_name)
    # code for task 2
    else:
        pass


if __name__ == "__main__":
    # start python program execution
    main()