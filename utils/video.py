import os

import cv2


def delete_video(file):
    return os.remove(file)
    
def make_temp_dir(file):
    # Split file name and drop extension
    basename = os.path.splitext(os.path.basename(file))[0]
    # Create subdir with file name
    temp_dir = os.path.join(file, basename)
    # Check if dir already exits
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)

def split_video(file):
    cap = cv2.VideoCapture(file)
    if cap.isOpened():
        success, frame = cap.read()
        images = []
        while success:
            images.append(frame)
            success, frame = cap.read()
    return images
    