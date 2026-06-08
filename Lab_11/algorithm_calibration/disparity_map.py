import cv2
import numpy as np
import matplotlib.pyplot as plt

from fast_reid.fastreid.data.transforms.autoaugment import brightness

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

# Calibrate each camera separately
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

# Stereo calibration + rectification
(RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
        objpoints, imgpointsLeft, imgpointsRight,
        K_left, D_left,
        K_right, D_right,
        image_size, None, None,
        cv2.CALIB_FIX_INTRINSIC,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

R2 = np.zeros([3, 3])
P1 = np.zeros([3, 4])
P2 = np.zeros([3, 4])
Q = np.zeros([4, 4])

(leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap) = cv2.fisheye.stereoRectify(
        K_left, D_left,
        K_right, D_right,
        image_size,
        rotationMatrix, translationVector,
        0, R2, P1, P2, Q,
        cv2.CALIB_ZERO_DISPARITY, (0, 0), 0, 0)

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, leftRectification, leftProjection, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, rightRectification, rightProjection, image_size, cv2.CV_16SC2)

# Disparity matchers
bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)
block = 5
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=block,
                             P1=8 * block * block, P2=32 * block * block)


def disparity(matcher, left, right):
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disp = matcher.compute(gl, gr).astype(np.float32) / 16.0
    return cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


titles = ["Block matching no calib", "Block matching calib",
          "Semi-Global Matching no calib", "Semi-Global Matching calib",
          "Original_L", "Original_R",
          "Undistorted_L", "Undistorted_R"]

fig, axes = plt.subplots(4, 2, figsize=(10, 10))
axes = axes.ravel()
current = [9]  # start at pair 09


def render(idx):
    img_l = cv2.imread(image_dir + "left_%02d.png" % idx)
    img_r = cv2.imread(image_dir + "right_%02d.png" % idx)
    if img_l is None or img_r is None:
        return

    # Rectify (after calibration)
    rect_l = cv2.remap(img_l, map1_left, map2_left, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map1_right, map2_right, cv2.INTER_LINEAR)

    # Disparity before (raw) and after (rectified) calibration
    images = [disparity(bm, img_l, img_r), disparity(bm, rect_l, rect_r),
              disparity(sgbm, img_l, img_r), disparity(sgbm, rect_l, rect_r),
              img_l, img_r, rect_l, rect_r]

    for i in range(8):
        axes[i].clear()
        if images[i].ndim == 2:          # disparity = grayscale
            axes[i].imshow(images[i], cmap="gray")
        else:                            # photo = color
            axes[i].imshow(rgb(images[i]))
        axes[i].set_title(titles[i])
    fig.suptitle("Pair %02d  (left/right arrow to change)" % idx)
    fig.canvas.draw_idle()


def on_key(event):
    if event.key == "right":
        current[0] = min(49, current[0] + 1)
    elif event.key == "left":
        current[0] = max(1, current[0] - 1)
    else:
        return
    render(current[0])


fig.canvas.mpl_connect("key_press_event", on_key)
render(current[0])
plt.tight_layout()
plt.show()

#comment of the result :
#close object big shift, big disparity appear bright on the image.
#distant object small shift, small disparity appear dark on the image.
