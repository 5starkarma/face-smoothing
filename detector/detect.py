# Adapted from the OpenCV.org courses.
# https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_opencv_dnn.py

import cv2

from utils.image import load_image, get_height_and_width


def detect_face(net, input_img, conf_threshold):
    """
    Detects face in an image.

    Parameters
    ----------
    net : dnn model
        cv2.dnn model
    input_file : np.array [H,W,3]
        Input BGR image
    conf_threshold : float
        Detection confidence threshold

    Returns
    -------
    detected_img : np.array [H,W,3]
        BGR image
    bboxes : list
        Bounding box coordinates
    """
    # Get height and width
    img_height, img_width = get_height_and_width(input_img)
    # Prepare image for net
    blob = cv2.dnn.blobFromImage(input_img, 1.0, (200, 200), [104, 117, 123], False, False)
    # Set the input for the net and run forward pass
    net.setInput(blob)
    detections = net.forward()
    # Given all detections
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # If detection is above threshold append to list
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * img_width)
            y1 = int(detections[0, 0, i, 4] * img_height)
            x2 = int(detections[0, 0, i, 5] * img_width)
            y2 = int(detections[0, 0, i, 6] * img_height)
            bboxes.append([x1, y1, x2, y2])
            # Draw bbox to image
            detected_img = cv2.rectangle(input_img.copy(), 
                                         (x1, y1), (x2, y2), 
                                         (0, 255, 0), 
                                         int(round(img_height / 150)), 8)
    return detected_img, bboxes



