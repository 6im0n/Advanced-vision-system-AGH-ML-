# Advanced Vision Systems — AGH

Coursework for the AGH Advanced Vision Systems (AVS) class. Labs cover
classical CV fundamentals (image ops, background subtraction, optical flow,
filtering, tracking); two projects apply those tools end-to-end.

## Contents

| Folder | Topic |
|---|---|
| `lab_1` | OpenCV basics — image I/O, drawing, color spaces |
| `lab_2` | Background subtraction via frame differencing (pedestrian sequence) |
| `lab_3` | Background subtraction with mean / median frame buffers (highway, office) |
| `lab_6` | Image filtering with SciPy / NumPy |
| `lab_7` | Optical flow — block-matching baseline |
| `lab_8` | Multi-object tracking with SiamFC appearance model |
| `project` | **Toothbrush bristle defect detection.** Classify SEM / high-res bristle-tip images as defective or clean. |
| `project_2` | **EVS-MOT pedestrian tracking.** Multi-object tracking on the EVS-MOT challenge dataset using BoT-SORT + a ConvNeXt-Small ReID encoder. Two pipeline variants — `yolo-version/` (own YOLOv8 detector + ensemble options) and `dettxt-version/` (uses the challenge-supplied `det.txt`). |

Each folder has its own `requirements.txt` / `pyvenv.cfg`.
Larger model weights are tracked via Git LFS.