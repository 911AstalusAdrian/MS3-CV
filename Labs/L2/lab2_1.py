'''
Write an application that loads two images:

a.      Scene image

b.      Logo image

And superposes the logo image over the scene and allows to see through the zones in the logo that do not contain details/information. Hint: use the opencv_logo.png as logo

You should show the problem running on a video flow with the logo over imposed on the video such that it does not hide the parts of the video where the logo is “transparent” (lacks info)
'''

# import cv2
# if __name__ == '__main__':

#     cam = cv2.VideoCapture(0)

#     cat = cv2.imread(, cv2.IMREAD_UNCHANGED)

#     cat = cv2.resize(cat, (200, 200))
#     mask = cat[:, :, 3] != 0

#     while True:
#         ret, frame = cam.read()
#         frame[0:200, 0:200][mask] = cat[:,:,:3][mask]

#         cv2.imshow('Camera', frame)

#         if cv2.waitKey(1) == ord('q'):
#             break

#     cam.release()
#     cv2.destroyAllWindows()


import cv2

cam = cv2.VideoCapture(0)
logo = cv2.imread(r'D:\Uni\MS3-CV\Labs\L2\opencv_logo.png', cv2.IMREAD_UNCHANGED)
print(logo.shape)
logo = cv2.resize(logo, (200, 200))
print(logo.shape) 

mask = logo[:, :, 3] != 0

while True:
    ret, frame = cam.read()

    frame[0:200, 0:200][mask] = logo[:,:,:3][mask]

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()