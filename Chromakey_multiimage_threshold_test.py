
def chorma_keyings(green_screen_imgs, scenic_img):
    """
    This function converts performs chroma keying techniques.
    Replaces green background screen from and image and replaces it
    with another background image.

    Args:
    - green_screen_img (cv2:img): cv2 image object of subject with greenscreen background
    - scenic_img (cv2:img): cv2 image object of background to put subject in
    """
    # Convert greenscreen image to HSB for better object separation
    img2 = cv.imread("greenScreen02.jpg")
    img3 = cv.imread("greenScreen03.jpg")
    img4 = cv.imread("greenScreen04.jpg")
    imgs = [green_screen_imgs, img2, img3, img4]
    for green_screen_img in imgs:
        green_screen_img, scenic_img = resize_to_larger(green_screen_img, scenic_img)
        green_screen_img_hsb = cv.cvtColor(green_screen_img, cv.COLOR_BGR2HSV)
        # green_screen_img_hsb = green_screen_img

        # Set thresholds for green color in terms of HSV chanels
        lower_green_thresholds = np.array([40, 80, 80])
        upper_green_thresholds = np.array([75, 255, 255])
        # Set a mask to isolate subject from green background
        mask = cv.inRange(green_screen_img_hsb, lower_green_thresholds, upper_green_thresholds)
        # display_image(mask)
        # exit()

        # Refine the mask using morphological operations to reduce noise
        kernel = np.ones((5,5),np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # display_image(mask)
        # exit()

        # Extract the subject using bitwise opeartions with inverse mask,
        # only the pixels with value 255, which is subject in inverse of mask will be retained
        subject = cv.bitwise_and(green_screen_img, green_screen_img, mask=~mask)
        # subject = np.copy(green_screen_img)
        # subject[mask != 0] = [0, 0, 0]
        # display_image(subject)

        # Extracted subject with white background
        subject_white_bg = subject + cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        # display_image(subject_white_bg)

        # Cutout subject layout on background and place subject on scenic background
        scenic_opening = np.copy(scenic_img)[0:scenic_img.shape[0], 0:scenic_img.shape[1]]
        scenic_opening[mask == 0] = [0, 0, 0]
        composite = scenic_opening + subject
        # display_image(scenic_opening)
        # display_image(composite)

        # Stack images horizontally
        top_row = np.hstack([green_screen_img, subject_white_bg])
        bottom_row = np.hstack([scenic_img, composite])

        # Stack images vertically
        final_img = np.vstack([top_row, bottom_row])

        # Display the final collage image
        display_image(final_img, "Chromakey")