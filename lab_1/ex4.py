import cv2
import numpy as np

image_1 = cv2.imread("lena.png")
image_2 = cv2.imread("mandrill.jpg")

gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# image 1 need to be resized
height, width = gray_1.shape[:2]
gray_2 = cv2.resize(gray_2, (width, height))

# add the two grayscale
added = cv2.add(gray_1, gray_2)
cv2.namedWindow('add', cv2.WINDOW_NORMAL)
cv2.imshow('add', added)

# sub and mult
subtracted = cv2.subtract(gray_1, gray_2)
cv2.namedWindow('sub', cv2.WINDOW_NORMAL)
cv2.imshow('sub', subtracted)

// convert before / normalise
multiplied = cv2.multiply(gray_1, gray_2)
cv2.namedWindow('multiplied', cv2.WINDOW_NORMAL)
cv2.imshow('multiplied', multiplied)

# comb linear
linear_combo = cv2.addWeighted(gray_1, 0.5, gray_2, 0.5, 0)
cv2.namedWindow('lin Combination', cv2.WINDOW_NORMAL)
cv2.imshow('lin Combination', linear_combo)

# difference
diff_manual = np.abs(gray_1.astype(np.float32) - gray_2.astype(np.float32)).astype(np.uint8)
# Opencv
diff_cv2 = cv2.absdiff(gray_1, gray_2)

cv2.namedWindow('absdiff manual' , cv2.WINDOW_NORMAL)
cv2.imshow('absdiff manual', diff_manual)

cv2.namedWindow('absdiff OpenCV', cv2.WINDOW_NORMAL)
cv2.imshow('absdiff OpenCV', diff_cv2)

# close the windows when keyboard touch
cv2.waitKey(0)
cv2.destroyAllWindows()
