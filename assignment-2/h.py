import cv2
import numpy as np
import sys


# Function to rescale an image. We keep the width fixed to 600 here. while maintaining aspect ratio.
def rescale_image_by_fixed_width(input_image, width=600):
    aspect_ratio = float(width) / input_image.shape[1]
    new_height = int(input_image.shape[0] * aspect_ratio)
    return cv2.resize(input_image, (width, new_height))


# Function to calculate chi-squared distance between two histograms
def get_chi_squared_distance(hist1, hist2):
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))


# Check if an image file is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python siftImages.py <image_file1> <image_file2> ...")
    sys.exit(1)

# If only one image file is provided, perform Task 1, which is only finding the number of key points of an Image
if len(sys.argv) == 2:
    # Get the image file path from the command-line argument
    image_path = sys.argv[1]

    # Load the image file
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Could not load the image.")
    else:
        # Rescale the image to VGA size
        rescaled_image = rescale_image_by_fixed_width(image, width=600)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect SIFT key-points
        keyPoints = sift.detect(gray_image, None)

        # Draw key-points on the rescaled image
        image_with_key_points = cv2.drawKeypoints(rescaled_image, keyPoints, outImage=None)

        # Display the original image and the image with highlighted key-points
        combined_image = np.hstack((rescaled_image, image_with_key_points))
        cv2.imshow("Original Image - Image with KeyPoints", combined_image)

        # Output the number of detected key-points
        num_key_points = len(keyPoints)
        print(f"# of key-points in {image_path} is {num_key_points}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# If multiple image files are provided, perform Task 2
else:
    # Initialize variables to store key-points and descriptors from all images
    all_key_points = []
    all_descriptors = []

    # Process each image provided as a command-line argument
    for image_path in sys.argv[1:]:
        # Load the image file
        image = cv2.imread(image_path)

        # Check if the image is loaded successfully
        if image is None:
            print(f"Error: Could not load the image {image_path}")
        else:
            # Rescale the image to VGA size
            rescaled_image = rescale_image_by_fixed_width(image, width=600)

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)

            # Initialize SIFT detector
            sift = cv2.SIFT_create()

            # Detect SIFT key-points and compute descriptors
            key_points, descriptors = sift.detectAndCompute(gray_image, None)

            # Append key-points and descriptors to the global lists
            all_key_points.extend(key_points)
            all_descriptors.extend(descriptors)

            # Output the number of key-points for the current image
            num_of_key_points = len(key_points)
            print(f"# of key-points in {image_path} is {num_of_key_points}")

    # Cluster the SIFT descriptors into K clusters (visual words)
    K_percentage = [0.05, 0.10, 0.20]
    K_values = [int(len(all_key_points) * percentage) for percentage in K_percentage]

    # Perform K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(np.array(all_descriptors, dtype=np.float32), K_values[0], None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)

    # Calculate histograms for each image
    histograms = []
    for i in range(len(sys.argv) - 1):
        image_descriptors = all_descriptors[i::len(sys.argv) - 1]  # Descriptors for the current image
        labels = labels[i::len(sys.argv) - 1]  # Labels for the current image
        hist, _ = np.histogram(labels, bins=range(K_values[0] + 1))
        histograms.append(hist)

    # Calculate dissimilarity matrices for different values of K
    for K, percentage in zip(K_values, K_percentage):
        print(f"\nK={percentage * 100}%*(total number of key-points)={K}")
        print("Dissimilarity Matrix")

        # Print column headers (image labels)
        image_labels = [chr(65 + i) for i in range(len(sys.argv) - 1)]
        header = "\t" + "\t".join(image_labels)
        print(header)

        dissimilarity_matrix = np.zeros((len(sys.argv) - 1, len(sys.argv) - 1))

        for i in range(len(sys.argv) - 1):
            for j in range(i, len(sys.argv) - 1):
                distance = get_chi_squared_distance(histograms[i], histograms[j])
                dissimilarity_matrix[i, j] = distance
                dissimilarity_matrix[j, i] = distance

        # Calculate the maximum and minimum values in the dissimilarity matrix
        max_distance = np.max(dissimilarity_matrix)
        min_distance = np.min(dissimilarity_matrix)

        # Normalize the dissimilarity matrix to the range [0, 1]
        normalized_matrix = (dissimilarity_matrix - min_distance) / (max_distance - min_distance + 1e-10)

        # Print the normalized dissimilarity matrix with proper formatting
        for i in range(len(sys.argv) - 1):
            row_label = image_labels[i]
            row_values = ["{:.2f}".format(value) for value in normalized_matrix[i]]
            row = f"{row_label}\t" + "\t".join(row_values)
            print(row)
