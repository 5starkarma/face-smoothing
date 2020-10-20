import cv2
import numpy as np


def smooth_face(cfg, box_face, bboxes):
    # Get Region Of Interest
    roi_img = box_face[bboxes[0][1]:bboxes[0][3], 
                       bboxes[0][0]:bboxes[0][2]]
    # Copy ROI
    temp_img = roi_img.copy()
    # Convert roi_img to HSV colorspace
    hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    # Get the mask for calculating histogram of the object and remove noise
    hsv_mask = cv2.inRange(hsv_img, 
                           np.array(cfg['image']['hsv_low']), 
                           np.array(cfg['image']['hsv_high']))
    # Make a 3 channel mask
    full_mask = cv2.merge((hsv_mask, hsv_mask, hsv_mask))

    # Apply blur on the created image
    blurred_img = cv2.bilateralFilter(roi_img, 9, 75, 75)
    # Apply mask to image
    masked_img = cv2.bitwise_and(blurred_img, full_mask)
    # Invert mask
    inverted_mask = cv2.bitwise_not(full_mask)
    # Created anti-mask
    masked_img2 = cv2.bitwise_and(temp_img, inverted_mask)
    # Add the masked images together
    full_img = cv2.add(masked_img2, masked_img)
    # Replace ROI on full image with blurred ROI
    box_face[bboxes[0][1]:bboxes[0][3], 
             bboxes[0][0]:bboxes[0][2]] = full_img
    return box_face, roi_img, full_mask, full_img
