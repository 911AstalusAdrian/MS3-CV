'''
Problem:
Write an algorithm for line and circle detection using the Hough method and apply it to images that contain potential lines and circles. 
Vary the arguments given to the algorithms and see the results.

Apply the line detection method and a segment detection method and a circle detection method.
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Canny Edge Detector
    return img, edges

# Hough Line Detection
def detect_lines(edges, img):
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    line_img = img.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_img

# Probabilistic Line Segment Detection
def detect_line_segments(edges, img):
    segments = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    segment_img = img.copy()
    if segments is not None:
        for x1, y1, x2, y2 in segments[:, 0]:
            cv2.line(segment_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return segment_img

# Circle Detection
def detect_circles(edges, img):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    circle_img = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            cv2.circle(circle_img, (x, y), r, (0, 0, 255), 2)
    return circle_img

# Main Function
def main(image_path):
    img, edges = preprocess_image(image_path)

    # Detect lines
    line_img = detect_lines(edges, img)

    # Detect line segments
    segment_img = detect_line_segments(edges, img)

    # Detect circles
    circle_img = detect_circles(edges, img)

    # Display results
    images = [img, edges, line_img, segment_img, circle_img]
    titles = ['Original Image', 'Edges', 'Hough Lines', 'Line Segments', 'Circles']

    for i in range(5):
        plt.subplot(2, 3, i + 1)
        if i == 1:  # Edge image in grayscale
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

image_path = r'D:\Uni\MS3-CV\Labs\L5\linesandcrcles.png'
main(image_path)