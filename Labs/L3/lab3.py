'''
Write an application that uses morphological operators to extract corners from an image using the following algorithm:
    1. R1 = Dilate(Img,cross)
    2. R1 = Erode(R1,Diamond)
    3. R2 = Dilate(Img,Xshape)
    4. R2 = Erode(R2,square)
    5. R = absdiff(R2,R1)
    6. Display(R)
Transform the input image to make it compatible with binary operators and display the results imposed over the original image. 
Apply the algorithm at least on the following images Rectangle and Building.
'''

import cv2
import numpy as np


image2 = r'D:\Uni\MS3-CV\Labs\L3\rectangle.png' 
image1 = r'D:\Uni\MS3-CV\Labs\L3\building.png'
original_img = cv2.imread(image1)
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to binary
_, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# Define structuring elements
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
diamond = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
xshape = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Step 1: Dilate and erode with cross and diamond
R1 = cv2.dilate(binary_img, cross)
R1 = cv2.erode(R1, diamond)

# Step 2: Dilate and erode with X-shape and square
R2 = cv2.dilate(binary_img, xshape)
R2 = cv2.erode(R2, square)

# Step 3: Compute the absolute difference
R = cv2.absdiff(R2, R1)

# Step 4: Overlay the results on the original image
corners = np.where(R > 0)
for (y, x) in zip(corners[0], corners[1]):
    cv2.circle(original_img, (x, y), radius=5, color=(0, 0, 255), thickness=2)

# Display results
cv2.imshow("Corners", R)
cv2.imshow("Overlay", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()