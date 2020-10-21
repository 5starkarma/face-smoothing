# Adapted from the OpenCV.org courses.
# https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_opencv_dnn.py

import cv2

from utils.image import load_image, get_height_and_width


def detect_face(net, input_file, conf_threshold):
    """
    Detects face in an image.

    Parameters
    ----------
    net : dnn model
        cv2.dnn model
    input_file : str
        Image file
    conf_threshold : float
        Detection confidence threshold

    Returns
    -------
    image : np.array [H,W,3]
        RGB image
    bboxes : list
        Bounding box cooridinates
    """
    # Load image
    img = load_image(input_file)
    # Get height and width
    img_height, img_width = get_height_and_width(img)
    # Prepare image for net
    blob = cv2.dnn.blobFromImage(img, 1.0, (200, 200), [104, 117, 123], False, False)
    # Set the input for the net
    net.setInput(blob)
    # Run a forward pass
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
            cv2.rectangle(img, (x1, y1), (x2, y2), 
                          (0, 255, 0), int(round(img_height / 150)), 8)
    return img, bboxes



