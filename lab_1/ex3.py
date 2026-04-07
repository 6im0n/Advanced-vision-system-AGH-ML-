import cv2

image = cv2.imread('mandrill.jpg')

cv2.imshow('Original', image)

height, width = image.shape[:2] # retrieving elements 0 and 1, i.e. height and width
scale = 2 # scale factor
Ix2 = cv2.resize(image, (int(scale * width), int(scale * height)))
cv2.namedWindow('Big Mandril', cv2.WINDOW_NORMAL)
cv2.imshow("Big Mandril", Ix2)


# close the windows when keyboard touch
cv2.waitKey(0)
cv2.destroyAllWindows()
