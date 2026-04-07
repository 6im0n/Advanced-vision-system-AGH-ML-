import cv2

image = cv2.imread('mandrill.jpg')

cv2.imshow('Original', image)

# place a point in the center  on the image using opencv and a rectangle next to it
h, w = image.shape[:2]
center_x, center_y = w // 2, h // 2
cv2.circle(image, (center_x +40, center_y), 10, (0, 255,0 ), -1)
cv2.rectangle(image, (center_x + 10, center_y - 25), (center_x + 60, center_y + 25), (0, 255, 0), 2)

# save the result in .png
cv2.namedWindow('result.png', cv2.WINDOW_NORMAL)
cv2.imwrite('result.png', image)
cv2.namedWindow('Processed img', cv2.WINDOW_NORMAL)
cv2.imshow('Processed img', image)

# close the windows when keyboard touch 
cv2.waitKey(0)
cv2.destroyAllWindows()

# make the same using m
