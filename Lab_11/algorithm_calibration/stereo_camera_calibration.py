import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

# inner size of chessboard
width = 9
height = 6
square_size = 0.025  # 0.025 meters

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((height * width, 1, 3), np.float64)
objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
objp = objp * square_size

# Arrays to store object points and image points for both cameras
objpoints = []
imgpointsLeft = []
imgpointsRight = []

img_width = 640
img_height = 480
image_size = (img_width, img_height)

path = "./dataset/"
image_dir = path + "pairs/"

# Detect chessboard corners in left and right images of every pair
number_of_images = 50
for i in range(1, number_of_images):
    img_l = cv2.imread(image_dir + "left_%02d.png" % i)
    img_r = cv2.imread(image_dir + "right_%02d.png" % i)
    if img_l is None or img_r is None:
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not (ret_l and ret_r):
        continue

    corners2_l = cv2.cornerSubPix(gray_l, corners_l, (3, 3), (-1, -1), criteria)
    corners2_r = cv2.cornerSubPix(gray_r, corners_r, (3, 3), (-1, -1), criteria)

    objpoints.append(objp)
    imgpointsLeft.append(corners2_l)
    imgpointsRight.append(corners2_r)

print("Found", len(objpoints), "valid image pairs for calibration")

# Calibrate each camera separately -> intrinsics K and distortion D
N_OK = len(objpoints)
K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))
K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

cv2.fisheye.calibrate(objpoints, imgpointsLeft, image_size, K_left, D_left, rvecs, tvecs, calibration_flags, criteria)
cv2.fisheye.calibrate(objpoints, imgpointsRight, image_size, K_right, D_right, rvecs, tvecs, calibration_flags, criteria)

imgpointsLeft = np.asarray(imgpointsLeft, dtype=np.float64)
imgpointsRight = np.asarray(imgpointsRight, dtype=np.float64)

(RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
        objpoints, imgpointsLeft, imgpointsRight,
        K_left, D_left,
        K_right, D_right,
        image_size, None, None,
        cv2.CALIB_FIX_INTRINSIC,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

R2 = np.zeros([3,3])
P1 = np.zeros([3,4])
P2 = np.zeros([3,4])
Q = np.zeros([4,4])

# Rectify calibration results
(leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap) = cv2.fisheye.stereoRectify(
        K_left, D_left,
        K_right, D_right,
        image_size,
        rotationMatrix, translationVector,
        0, R2, P1, P2, Q,
        cv2.CALIB_ZERO_DISPARITY, (0,0) , 0, 0)

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, leftRectification,
        leftProjection, image_size, cv2.CV_16SC2)

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, rightRectification,
        rightProjection, image_size, cv2.CV_16SC2)

# Load a test pair to rectify
img_l = cv2.imread(image_dir + "left_09.png")
img_r = cv2.imread(image_dir + "right_09.png")

dst_L = cv2.remap(img_l, map1_left, map2_left, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_r, map1_right, map2_right, cv2.INTER_LINEAR)

N, XX, YY = dst_L.shape[::-1] # RGB image size

visRectify = np.zeros((YY, XX*2, N), np.uint8) # create a new image with a new size (height, 2*width)
visRectify[:,0:XX:,:] = dst_L      # left image assignment
visRectify[:,XX:XX*2:,:] = dst_R   # right image assignment

# draw horizontal lines
for y in range(0,YY,10):
    cv2.line(visRectify, (0,y), (XX*2,y), (255,0,0))

cv2.imshow('visRectify',visRectify)  # display image with lines
cv2.waitKey(0)
cv2.destroyAllWindows()