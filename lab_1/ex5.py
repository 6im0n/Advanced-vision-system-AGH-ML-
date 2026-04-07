import cv2
import numpy as np

image_1 = cv2.imread("lena.png")

gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

# display the histogram of the image
def hist(img):
    h = np.zeros((256, 1), np.float32)  # creates and zeros single - column arrays
    height, width = img.shape[:2]  # shape - we take the first 2 values
    for y in range(height):
        for x in range(width):
            h[img[y, x]] += 1
    return h

def display_hist(h, name):
    canvas = np.full((300, 256), 255, dtype=np.uint8)
    h_norm = np.int32(np.around(255 * h / h.max())).flatten()
    for x, y in enumerate(h_norm):
        cv2.line(canvas, (x, 299), (x, 299 - y), (0, 0, 0))
    cv2.imshow(name, canvas)

cv2.namedWindow('gray_1', cv2.WINDOW_NORMAL)
cv2.imshow('gray_1', gray_1)

# Calculate and display histogram
h1 = hist(gray_1)
display_hist(h1, 'histogram_1')

#------------------------histogram equalization-----------------------

#classical historgram equalization
IGE = cv2.equalizeHist(gray_1)
cv2.namedWindow('classical histo', cv2.WINDOW_NORMAL)
cv2.imshow('classical histo', IGE)

#clache histo
clahe = cv2 . createCLAHE (clipLimit =2.0,tileGridSize =(8,8))
# clipLimit - maximum height of the histogram bar - values above are distributed among neighbours
# tileGridSize - size of a single image block ( local method , operates on separate image blocks )
I_CLAHE = clahe.apply(gray_1)
cv2.namedWindow('clahe histo', cv2.WINDOW_NORMAL)
cv2.imshow('clahe histo', I_CLAHE)


# close the windows when keyboard touch
cv2.waitKey(0)
cv2.destroyAllWindows()
