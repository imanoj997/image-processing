# importing necessary python packages
import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def resize_to_vga(img):
    """
    This function resizes the image to be roughly VGA size (480x600) while maintaining its aspect ratio.

    Args:
    - img (cv:img): image to be resized

    Returns:
    - Resized image (cv:img)
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
    - image (cv: img): image to be displayed
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

def sift_extractor(image, extract_des=False):
    """
    Extracts and returns SIFT keypoints and descriptors from an image

    Args:
    - image (cv: img): Image from which to extract SIFT keypoints and descriptors
    - extract_des (boolean): to extract descriptors or not - default fasle
    """
    # Initialize the SIFT detector
    sift = cv.SIFT_create()
    if extract_des:
        # Extract SIFT keypoints from the Y component
        kp = sift.detect(image, None)
        des = None
    else:
        # Extract SIFT keypoints and descriptors from the Y component
        kp, des = sift.detectAndCompute(image, None)

    return kp, des


def display_sift_features(image, image_name):
    """
    Extracts SIFT keypoints from the luminance (Y) component of an image 
    and displays the image with the keypoints highlighted.

    Args:
    - image (cv: img): Image from which to extract SIFT keypoints.
    - image_name (str): name of the image
    """
    working_image = image.copy()
    yuv = cv.cvtColor(working_image, cv.COLOR_BGR2YUV)

    # Extract the Y (luminance) component from the YUV image
    y_channel = yuv[:,:,0]

    # Detect SIFT keypoints from the Y component
    kp = sift_extractor(y_channel)[0]

    # Loop through each detected keypoint to draw a cross
    for keypoint in kp:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        line_length = 5
        cv.line(working_image, (x - line_length, y), (x + line_length, y), (0, 0, 255), 1)
        cv.line(working_image, (x, y - line_length), (x, y + line_length), (0, 0, 255), 1)

    # Draw the keypoints with circles and orientation lines on the working image
    image_w_keypoints = cv.drawKeypoints(y_channel, kp, working_image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Stack the original image and the image with keypoints side-by-side for comparison
    final_image = np.hstack([image, image_w_keypoints])
    display_image(final_image)
    print(f"# of keypoints in {image_name} is {len(kp)}")


def compute_chi_square_distance(his1, his2):
    """
    Calculates chi-squared distance between two histograms

    - his1: first histogram
    - his2: second histogram
    """
    return 0.5 * np.sum(((his1 - his2) ** 2) / (his1 + his2 + 1e-10))


def compute_k_values(descriptors, percentages):
    return [int(len(descriptors) * perc) for perc in percentages]


def perform_kmeans_clustering(descriptors, k_val):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, lbls, centers = cv.kmeans(descriptors.astype(np.float32), k_val, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    return lbls


def compute_histograms(descriptors, lbls, count, k_val):
    hists = []
    for idx in range(count):
        current_desc = descriptors[idx::count]
        current_labels = lbls[idx::count]
        hist_data, _ = np.histogram(current_labels, bins=range(k_val + 1))
        hists.append(hist_data)
    return hists


def print_dissimilarity_matrix(matrix, labels):
    print("\t" + "\t".join(labels))
    for i, values in enumerate(matrix):
        formatted_values = ["{:.2f}".format(val) for val in values]
        print(f"{labels[i]}\t" + "\t".join(formatted_values))


def display_dissimilarity_matrix(keypoints_list, descriptors_list):
    """
    When given an array of images, combine descriptors from all images,
    applies k-means clustering and computes dissimilarity matrix for all images

    - keypoints_list (array): array of keypoints from all images
    - descriptors_list (array): array of descriptors from all images
    """
    percentages = [0.05, 0.10, 0.20]
    all_k_values = compute_k_values(keypoints_list, percentages)

    labels = perform_kmeans_clustering(np.array(descriptors_list), all_k_values[0])

    image_count = len(sys.argv) - 1
    hists = compute_histograms(descriptors_list, labels, image_count, all_k_values[0])

    for K, perc in zip(all_k_values, percentages):
        print(f"\nK={perc*100}%*(total number of key-points)={K}")
        print("Dissimilarity Matrix")

        image_tags = [chr(65 + i) for i in range(image_count)]
        matrix = np.zeros((image_count, image_count))

        for i in range(image_count):
            for j in range(i, image_count):
                dist = compute_chi_square_distance(hists[i], hists[j])
                matrix[i, j] = dist
                matrix[j, i] = dist

        max_val = np.max(matrix)
        min_val = np.min(matrix)
        norm_matrix = (matrix - min_val) / (max_val - min_val + 1e-10)

        print_dissimilarity_matrix(norm_matrix, image_tags)

def main():
    if len(sys.argv) == 1:
        print("Error: Invalid command \n Use 'python siftImages.py image-path' to run task 1 \n \
        Use 'python siftImages.py image-path1 image-path2.....' to run task 2")
        return

    # Code for task 1 in case of 1 image as argument
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
            display_sift_features(resized_image, image_name)
    # Code for task 1 in case of more tgan 1 images as argument
    else:
        image_path_array = sys.argv[1:]

        # Intialize arrays to store keypoints and descriptors for future use
        keypoints_list = []
        descriptors_list = []

        for image_path in image_path_array:
            # Read the image
            image = cv.imread(image_path)
            resized_image = resize_to_vga(image) # resize the image close to VGA size
            image_name = os.path.basename(image_path)

            working_image = resized_image.copy()
            yuv = cv.cvtColor(working_image, cv.COLOR_BGR2YUV)

            # Extract the Y (luminance) component from the YUV image
            y_channel = yuv[:,:,0]

            kp, des = sift_extractor(y_channel) # Extract keypoints and descriptors

            # Add keypoints and descriptors to an array
            keypoints_list += [k for k in kp]
            descriptors_list += [d for d in des]

            print(f"# of keypoints in {image_name} is {len(kp)}")
        display_dissimilarity_matrix(keypoints_list, descriptors_list)


if __name__ == "__main__":
    # start python program execution
    main()