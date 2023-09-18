# Callback function for trackbar (does nothing in this example)
def nothing(x):
    pass

# Load your green screen image
hsv = cv.cvtColor(green_screen_img, cv.COLOR_BGR2HSV)

# Create a window
cv.namedWindow('image')

# Create trackbars for Hue, Saturation, and Value
cv.createTrackbar('Hue Lower', 'image', 0, 255, nothing)
cv.createTrackbar('Hue Upper', 'image', 0, 255, nothing)

cv.createTrackbar('Sat Lower', 'image', 0, 255, nothing)
cv.createTrackbar('Sat Upper', 'image', 0, 255, nothing)

cv.createTrackbar('Val Lower', 'image', 0, 255, nothing)
cv.createTrackbar('Val Upper', 'image', 0, 255, nothing)

while(1):
    # Get the position of the trackbars
    h_lower = cv.getTrackbarPos('Hue Lower', 'image')
    h_upper = cv.getTrackbarPos('Hue Upper', 'image')
    s_lower = cv.getTrackbarPos('Sat Lower', 'image')
    s_upper = cv.getTrackbarPos('Sat Upper', 'image')
    v_lower = cv.getTrackbarPos('Val Lower', 'image')
    v_upper = cv.getTrackbarPos('Val Upper', 'image')

    # Threshold the HSV image
    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(green_screen_img, green_screen_img, mask=mask)

    cv.imshow('image', res)
    k = cv.waitKey(1) & 0xFF
    if k == 27:  # Press 'ESC' to exit
        break

cv.destroyAllWindows()