import cv2

image = cv2.imread('mandrill.jpg')

cv2.imshow('Original', image)

IHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
IG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('IHSV', cv2.WINDOW_NORMAL)
cv2.imshow('IHSV', IHSV)
cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
cv2.imshow('Grayscale', IG)

#display the H,S,V component of the iamge independlatly
IH = IHSV [: ,: ,0]
IS = IHSV [: ,: ,1]
IV = IHSV [: ,: ,2]
cv2.namedWindow('Hue', cv2.WINDOW_NORMAL)
cv2.imshow('Hue', IH)
cv2.namedWindow('Saturation', cv2.WINDOW_NORMAL)
cv2.imshow('Saturation', IS)
cv2.namedWindow('Value', cv2.WINDOW_NORMAL)
cv2.imshow('Value', IV)


# close the windows when keyboard touch 
cv2.waitKey(0)
cv2.destroyAllWindows()