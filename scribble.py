import cv2
import numpy as np

img_name = '0184_BP_H_-1_A_211_1595085939000_compressed.JPG'

# Create a black image with the same size as the input image
def create_blank_image(img):
    return np.zeros((img.shape[0], img.shape[1]), np.uint8)

# Mouse callback function to draw black pixels on the image
def draw_on_image(event, x, y, flags, param):
    global img, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Load the input image
img = cv2.imread(img_name)

# Create a blank image with the same size as the input image
blank_img = create_blank_image(img)

# Create a window to display the input image
cv2.namedWindow('Input Image')
cv2.imshow('Input Image', img)

# Set up the mouse callback function
drawing = False
cv2.setMouseCallback('Input Image', draw_on_image)

# Wait for the user to draw on the image
while True:
    cv2.imshow('Input Image', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Convert the drawn image to grayscale and threshold it
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

# Save the thresholded image as the output image
cv2.imwrite(img_name[:-4]+'_scribble.png', thresh_img)

# Create a window to display the output image
cv2.namedWindow('Output Image')
cv2.imshow('Output Image', blank_img)

# Wait for the user to close the windows
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
