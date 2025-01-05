'''
Implement the fill contour using morphological operations algorithm
'''

import cv2
import numpy as np

def fill_contour_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    seed_point = (img.shape[0] // 2, img.shape[1] // 2) # starting point: center of the image

    if img is None:
        raise ValueError("Image not found or could not be loaded.")

    # Convert to binary (thresholding)
    _, A = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    A = cv2.bitwise_not(A)
    # Initialize the seed point
    X_k = np.zeros_like(A, dtype=np.uint8)
    X_k[seed_point[1], seed_point[0]] = 255  # Set seed point (x, y)

    # Structuring element: 3x3 square
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    while True:
        # Morphological dilation
        X_k1 = cv2.dilate(X_k, kernel)  # X_k ⊕ B
        
        X_k1 = cv2.bitwise_and(X_k1, cv2.bitwise_not(A))  # (X_k ⊕ B) ∩ A^C
        
        if np.array_equal(X_k1, X_k):
            break
        X_k = X_k1

    # Combine the filled area with the original contour
    
    filled_object = cv2.bitwise_or(A, X_k)
    A = cv2.bitwise_not(A)
    filled_object = cv2.bitwise_not(filled_object)
    return A, filled_object

# Example usage:
if __name__ == "__main__":
    image_path = r'D:\Uni\MS3-CV\Labs\L2\circle.png'
    seed_point = (252, 252)

    # Process the image
    try:
        contour, filled = fill_contour_image(image_path)

        # Show results
        cv2.imshow("Original Contour", contour)
        cv2.imshow("Filled Contour", filled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ValueError as e:
        print(e)