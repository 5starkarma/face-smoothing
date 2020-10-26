import os

import cv2

from .image import check_if_adding_bboxes


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
    

def process_video(file, output_dir, cfg, net):
    """
    Splits video into frames then processes each frame individually
    before merging all the frames back into a video.

    Parameters
    ----------
    file : H.264 video
        Input video

    output_dir : str
        Output directory where processed video will be saved

    cfg : dict
        Dictionary of configurations

    net : Neural Network object
        Pre-trained model ready for foward pass

    bboxes : list [[x1, y1, x2, y2],...]
        List of lists of bbox coordinates

    Returns
    -------
    images : tuple
        Tuple of BGR images
    """
    # Split video into frames
    images = split_video(file)
    # Add brackets and extension to filename
    filename = os.path.join(output_dir, cfg['video']['output']) + '{}.mp4'
    # If a file of this name exists increase the counter by 1
    counter = 0
    while os.path.isfile(filename.format(counter)):
        counter += 1
    # Apply counter to filename
    output_path = filename.format(counter)
    # Get height and width of 1st image
    height, width, _  = check_img_size(images[0]).shape
    # Create VideoWriter object
    video = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'FMP4'), 
                            30, 
                            (width,height))
    for image in images:
        # Process frames
        _, _, _, _, _, output_w_bboxes, output_img = process_image(image, cfg, net)
        # Check for --show-detections flag
        output_img = check_if_adding_bboxes(args, img_steps)        
    # Release video writer object
    video.release()

