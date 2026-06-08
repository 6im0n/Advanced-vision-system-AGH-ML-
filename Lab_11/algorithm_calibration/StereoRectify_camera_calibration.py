import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

# inner size of chessboard
width = 9
height = 6
square_size = 0.025  # 0.025 meters

# prepare object points (N, 1, 3) -> matches corner shape (N, 1, 2)
objp = np.zeros((height * width, 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
objp = objp * square_size

# Arrays to store object points and image points for both cameras
objpoints = []
imgpoints_l = []
imgpoints_r = []

img_width = 640
img_height = 480
image_size = (img_width, img_height)

path = "./dataset/"
image_dir = path + "pairs/"

#Detect chessboard corners in left and right images of every pair

number_of_images = 50
for i in range(1, number_of_images):
    img_name_l = "left_%02d.png" % i
    img_name_r = "right_%02d.png" % i

    img_l = cv2.imread(image_dir + img_name_l)
    img_r = cv2.imread(image_dir + img_name_r)

    if img_l is None or img_r is None:
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if not (ret_l and ret_r):
        continue

    # Skip pairs where corners sit too close to the image border
    Y, X = gray_l.shape
    border_threshold_x = X / 12
    border_threshold_y = Y / 12

    bad_pair = False
    for p in (corners_l, corners_r):
        if p[:, :, 0].min() < border_threshold_x or p[:, :, 0].max() > X - border_threshold_x or \
           p[:, :, 1].min() < border_threshold_y or p[:, :, 1].max() > Y - border_threshold_y:
            bad_pair = True
            break
    if bad_pair:
        continue

    corners2_l = cv2.cornerSubPix(gray_l, corners_l, (3, 3), (-1, -1), criteria)
    corners2_r = cv2.cornerSubPix(gray_r, corners_r, (3, 3), (-1, -1), criteria)

    objpoints.append(objp)
    imgpoints_l.append(corners2_l)
    imgpoints_r.append(corners2_r)

    cv2.drawChessboardCorners(img_l, (width, height), corners2_l, ret_l)
    cv2.namedWindow("Corners Left", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Corners Left", 1400, 700)
    cv2.imshow("Corners Left", img_l)
    cv2.namedWindow("Corners Right", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Corners Right", 1400, 700)
    cv2.imshow("Corners Right", img_r)
    cv2.waitKey(5)

cv2.destroyAllWindows()
print(f"Found {len(objpoints)} valid image pairs for calibration")

# Calibrate each camera separately -> intrinsics K and distortion D
N_OK = len(objpoints)
K_l = np.zeros((3, 3))
D_l = np.zeros((4, 1))
K_r = np.zeros((3, 3))
D_r = np.zeros((4, 1))
rvecs_l = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
tvecs_l = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
rvecs_r = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
tvecs_r = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

rms_l, K_l, D_l, rvecs_l, tvecs_l = cv2.fisheye.calibrate(
    objpoints, imgpoints_l, image_size, K_l, D_l, rvecs_l, tvecs_l, calibration_flags, criteria)
rms_r, K_r, D_r, rvecs_r, tvecs_r = cv2.fisheye.calibrate(
    objpoints, imgpoints_r, image_size, K_r, D_r, rvecs_r, tvecs_r, calibration_flags, criteria)

print("Left  RMS =", rms_l)
print("K_left =\n", K_l)
print("D_left =\n", D_l.ravel())
print("Right RMS =", rms_r)
print("K_right =\n", K_r)
print("D_right =\n", D_r.ravel())


#  Uncalibrated rectification
# Distortion must be removed first, so the fundamental matrix and the
# rectifying homographies are computed in the *undistorted* pixel space.

all_l = np.concatenate(imgpoints_l, axis=0)  # (total, 1, 2)
all_r = np.concatenate(imgpoints_r, axis=0)

# Undistort corner points into the undistorted image (P=K keeps pixel coords)
und_l = cv2.fisheye.undistortPoints(all_l, K_l, D_l, R=np.eye(3), P=K_l).reshape(-1, 2).astype(np.float32)
und_r = cv2.fisheye.undistortPoints(all_r, K_r, D_r, R=np.eye(3), P=K_r).reshape(-1, 2).astype(np.float32)

# Fundamental matrix from undistorted matches, then keep only inliers
F, mask = cv2.findFundamentalMat(und_l, und_r, cv2.FM_LMEDS)
mask = mask.ravel().astype(bool)
und_l = und_l[mask]
und_r = und_r[mask]

# Rectifying homographies (Hartley's uncalibrated method)
ret, H1, H2 = cv2.stereoRectifyUncalibrated(und_l, und_r, F, image_size)
print("H1 =\n", H1)
print("H2 =\n", H2)

#Undistort a test pair, rectify, merge and draw horizontal lines
img_l_test = cv2.imread(image_dir + "left_09.png")
img_r_test = cv2.imread(image_dir + "right_09.png")

# Remove fisheye distortion (R=eye, P=K -> stay in original pixel frame)
map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(K_l, D_l, np.eye(3), K_l, image_size, cv2.CV_16SC2)
map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(K_r, D_r, np.eye(3), K_r, image_size, cv2.CV_16SC2)
undist_l = cv2.remap(img_l_test, map1_l, map2_l, cv2.INTER_LINEAR)
undist_r = cv2.remap(img_r_test, map1_r, map2_r, cv2.INTER_LINEAR)

# Show distortion removal result before rectification
undist_pair = np.hstack((undist_l, undist_r))
cv2.namedWindow("Undistorted (no rectify)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Undistorted (no rectify)", 1600, 700)
cv2.imshow("Undistorted (no rectify)", undist_pair)

# Apply rectifying homographies to the undistorted images
rect_l = cv2.warpPerspective(undist_l, H1, image_size)
rect_r = cv2.warpPerspective(undist_r, H2, image_size)

combined = np.hstack((rect_l, rect_r))

# Draw horizontal lines: corresponding points should now lie on the same rows
for y in range(0, combined.shape[0], 25):
    cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

cv2.namedWindow("Rectified Stereo Pair", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rectified Stereo Pair", 1600, 700)
cv2.imshow("Rectified Stereo Pair", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()