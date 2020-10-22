import cv2
import numpy as np


def get_roi(detected_img, bboxes, box_num):
    """
    Crop detected image to size of detection

    Parameters
    ----------
    detected_img : np.array [H,W,3]
        BGR image
    bboxes : 
    """
    return detected_img[bboxes[box_num][1]:bboxes[box_num][3], 
                        bboxes[box_num][0]:bboxes[box_num][2]]


def smooth_face(cfg, detected_img, bboxes):
    """
    Smooth faces in an image using bilateral filtering.

    Parameters
    ----------
    cfg : dict
        Dictionary of configurations
    box_face : np.array [H,W,3]
        BGR image
    bboxes : list
        List of detected bounding boxes

    Returns
    -------
    detected_img : np.array [H,W,3]
        BGR image with face detections
    roi : np.array [H,W,3]
        BGR image
    full_mask : np.array [H,W,3]
        BGR image
    full_img : np.array [H,W,3]
        BGR image
    """
    # Get Region Of Interest of each face
    for box_num in range(len(bboxes)):
        print(f'Face detected: {bboxes[box_num]}')
        # Get Region of Interest
        roi = get_roi(detected_img, bboxes, box_num)
        # Copy ROI
        temp_img = roi.copy()
        # Convert roi_img to HSV colorspace
        hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Get the mask for calculating histogram of the object and remove noise
        hsv_mask = cv2.inRange(hsv_img, 
                               np.array(cfg['image']['hsv_low']), 
                               np.array(cfg['image']['hsv_high']))
        # Make a 3 channel mask
        full_mask = cv2.merge((hsv_mask, hsv_mask, hsv_mask))
        # Apply blur on the created image
        blurred_img = cv2.bilateralFilter(roi, 
                                          cfg['filter']['diameter'], 
                                          cfg['filter']['sigma_1'], 
                                          cfg['filter']['sigma_2'])
        # Apply mask to image
        masked_img = cv2.bitwise_and(blurred_img, full_mask)
        # Invert mask
        inverted_mask = cv2.bitwise_not(full_mask)
        # Created anti-mask
        masked_img2 = cv2.bitwise_and(temp_img, inverted_mask)
        # Add the masked images together
        full_img = cv2.add(masked_img2, masked_img)
        # Init smoothed image
        smoothed_img = detected_img.copy()
        # Replace ROI on full image with blurred ROI
        smoothed_img[bboxes[box_num][1]:bboxes[box_num][3], 
                     bboxes[box_num][0]:bboxes[box_num][2]] = full_img
    return smoothed_img, roi, full_mask, full_img
