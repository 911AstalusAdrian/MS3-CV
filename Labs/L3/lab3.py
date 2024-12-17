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
import matplotlib.pyplot as plt

def extract_corners(image_path):
    """
    Extract corners from an image using morphological operations.

    Args:
    - image_path: str, path to the input image.

    Returns:
    - None, displays the processed results.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(image_path)
    
    # Transform the image to binary
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Structuring elements
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    diamond = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)
    x_shape = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=np.uint8)
    square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Step 1: R1 = Dilate(Img, cross)
    R1 = cv2.dilate(binary_img, cross)
    # Step 2: R1 = Erode(R1, Diamond)
    R1 = cv2.erode(R1, diamond)
    
    # Step 3: R2 = Dilate(Img, Xshape)
    R2 = cv2.dilate(binary_img, x_shape)
    # Step 4: R2 = Erode(R2, square)
    R2 = cv2.erode(R2, square)
    
    # Step 5: R = absdiff(R2, R1)
    R = cv2.absdiff(R2, R1)
    
    # Superimpose the corners on the original image
    corners = cv2.bitwise_and(original, original, mask=R)
    
    # Display the results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Binary Image")
    plt.imshow(binary_img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("R1 (Dilate + Erode)")
    plt.imshow(R1, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("R2 (Dilate + Erode)")
    plt.imshow(R2, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("R (Corner Detection)")
    plt.imshow(R, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Corners on Original")
    plt.imshow(cv2.cvtColor(corners, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

# Apply the algorithm on Rectangle and Building images
print("Processing Rectangle image...")
extract_corners(r'D:\Uni\MS3-CV\Labs\L3\rectangle.png')  

print("Processing Building image...")
extract_corners(r'D:\Uni\MS3-CV\Labs\L3\building.png')  