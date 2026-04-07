import cv2
import numpy as np

image_1 = cv2.imread("lena.png")


gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('gray_1', cv2.WINDOW_NORMAL)
cv2.imshow('gray_1', gray_1)


blurred_image = cv2.GaussianBlur(gray_1, (5, 5), 0)
cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
cv2.imshow("Gaussian Blur", blurred_image)

#sobel
sobel_x = cv2.Sobel(gray_1, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_1, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
cv2.namedWindow('sobel', cv2.WINDOW_NORMAL)
cv2.imshow("sobel", np.uint8(sobel_combined))

#laplacian filter
laplacian = cv2.Laplacian(gray_1, cv2.CV_64F)
cv2.namedWindow('Laplacian', cv2.WINDOW_NORMAL)
cv2.imshow("Laplacian", np.uint8(np.absolute(laplacian)))

# median filtration
median_blurred = cv2.medianBlur(gray_1, 5)
cv2.namedWindow('Meedian', cv2.WINDOW_NORMAL)
cv2.imshow("Meedian", median_blurred)

# close the windows when keyboard touch
cv2.waitKey(0)
cv2.destroyAllWindows()
