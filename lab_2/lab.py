import cv2
import os
import numpy as np

def load_sequence(folder_path, roi_file):
    # Read start and end points from temporalroii
    with open(roi_file, 'r') as f:
        line = f.readline()
        start_frame, end_frame = map(int, line.split())

    print(f"Loading sequence from frame {start_frame} to {end_frame}")

    prev_img = None
    for i in range(start_frame, end_frame + 1):
        filename = f"in{i:06d}.jpg"
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)
        print(f"Load image: {filename}")
        if img is None:
            print(f"Failed to load image: {file_path}")
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_img is not None:
            gray_int = gray.astype('int')
            prev_img_int = prev_img.astype('int')
            # frame subtraction and binarisation
            diff = cv2.absdiff(gray_int, prev_img_int)
            diff = diff.astype('uint8')
            # to avoid problem we neeed to make subtraction and perform a conversion to uint8..
            (T, thresh) = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

            # remove noise using gaussian and median filter
            thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
            thresh = cv2.medianBlur(thresh, 3)

            cv2.imshow('Foreground Mask', thresh)
            cv2.moveWindow('Foreground Mask', 0, 0)
            # Perform filtering using erosion and dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.erode(thresh, kernel, iterations=3)
            thresh = cv2.dilate(thresh, kernel, iterations=3)

            cv2.imshow('Foreground Mask filtering', thresh)
            cv2.moveWindow('Foreground Mask filtering', 600, 0)

            # connected components
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

            if retval > 0:
                cv2.imshow(" Labels ", np.uint8(labels / retval * 255))
                cv2.moveWindow(" Labels ", 1200, 0)

            # display bounding boxes for ALL detected objects.
            I_VIS = img.copy()  # copy of the input image
            min_area = 200  # Optional: minimum area to consider (adjust as needed to filter noise)
            for i in range(1, retval):  # Skip background (index 0)
                area = stats[i, 4]
                if area >= min_area:  # Only process objects above minimum area
                    # Drawing a bbox for each object
                    x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
                    cv2.rectangle(I_VIS, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Optional: print area and component index on the box
                    cv2.putText(I_VIS, f"Area: {area}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                    cv2.putText(I_VIS, f"ID: {i}", (int(centroids[i, 0]), int(centroids[i, 1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv2.imshow('Result', I_VIS)
            cv2.moveWindow('Result', 0, 600)

            # Metrics computation (P, R, F1)
            # Assuming 'gray' here is actually the ground truth mask for comparison
            # In a real scenario, one would compare the 'thresh' (prediction) with a GT image
            gt_mask = gray # Using current gray as dummy GT for structure
            tp = np.sum(np.logical_and(thresh == 255, gt_mask == 255))
            fp = np.sum(np.logical_and(thresh == 255, gt_mask == 0))
            fn = np.sum(np.logical_and(thresh == 0, gt_mask == 255))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Frame {i} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")


        prev_img = gray
        cv2.imshow('Sequence', img)
        cv2.moveWindow('Sequence', 600, 600)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sequence_folder = "./pedestrian/input"
    roi_path = "./pedestrian/temporalROI.txt"  # Assuming the file is in the working directory or provide full path
    
    if os.path.exists(sequence_folder):
        load_sequence(sequence_folder, roi_path)
    else:
        print(f"Directory not found: {sequence_folder}")
