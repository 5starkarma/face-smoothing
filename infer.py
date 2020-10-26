import os
import argparse
import yaml
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt

from detector.detect import detect_face
from detector.smooth import smooth_face
from utils.image import (load_image, 
                         save_image, 
                         save_steps, 
                         check_img_size,
                         get_height_and_width)
from utils.video import split_video
from utils.types import (is_image,
                         is_video,
                         is_directory)


def parse_args():
    """
    Argument parser for cli.

    Returns
    -------
    args : ArgumentParser object
        Contains all the cli arguments
    """
    parser = argparse.ArgumentParser(description='Facial detection and \
                                     smoothing using OpenCV.')
    parser.add_argument('--input', 
                        type=str, 
                        help='Input file or folder',
                        default='data/images/hillary_clinton.jpg')
    parser.add_argument('--output', 
                        type=str, 
                        help='Output file or folder',
                        default='data/output')
    parser.add_argument('--show-detections', 
                        action='store_true',
                        help='Displays bounding boxes during inference.')
    parser.add_argument('--save-steps', 
                        action='store_true',
                        help='Saves each step of the image.')
    args = parser.parse_args()
    # assert args.image_shape is None or len(args.image_shape) == 2, \
    #     'You need to provide a 2-dimensional tuple as shape (H,W)'
    # assert (is_image(args.input) and is_image(args.output)) or \
    #        (not is_image(args.input) and not is_image(args.input)), \
    #     'Input and output must both be images or folders'
    return args


def load_configs():
    """
    Loads the project configurations.

    Returns
    -------
    configs : dict
        A dictionary containing the configs
    """
    with open('/content/drive/My Drive/Colab Notebooks/face-smoothing'\
               '/configs/configs.yaml', 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def draw_bboxes(output_img, cfg, bboxes):
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    output_img : np.array [H,W,3]
        BGR image of face

    cfg : dict
        Dictionary of configurations

    bboxes : list [[x1, y1, x2, y2],...]
        List of lists of bbox coordinates

    Returns
    -------
    image : np.array [H,W,3]
        BGR image with bounding boxes
    """
    # Create copy of image
    output_w_bboxes = output_img.copy()
    # Get height and width
    img_height, img_width = get_height_and_width(output_w_bboxes)
    # Draw bboxes
    for i in range(len(bboxes)):
        top_left = (bboxes[i][0], bboxes[i][1])
        btm_right = (bboxes[i][2], bboxes[i][3])
        cv2.rectangle(output_w_bboxes, 
                      top_left, 
                      btm_right, 
                      cfg['image']['bbox_color'], 
                      2)
    return output_w_bboxes        


def process_image(input_img, cfg, net):
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    output_img : np.array [H,W,3]
        BGR image of face

    cfg : dict
        Dictionary of configurations

    bboxes : list [[x1, y1, x2, y2],...]
        List of lists of bbox coordinates

    Returns
    -------
    images : tuple
        Tuple of BGR images
    """
    # Make sure image is less than 1081px wide
    input_img = check_img_size(input_img)
    # Detect face
    detected_img, bboxes = detect_face(cfg, net, input_img)
    # Smooth face and return steps
    output_img, roi_img, hsv_mask, smoothed_roi = smooth_face(cfg, 
                                                              input_img, 
                                                              bboxes)
    # Draw bboxes on output_img
    output_w_bboxes = draw_bboxes(output_img, cfg, bboxes)
    return (input_img, detected_img, roi_img, hsv_mask, 
            smoothed_roi, output_w_bboxes, output_img)


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
    input_img = check_img_size(images[0])
    height, width, _ = input_img.shape  
    # Create VideoWriter object
    video = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'FMP4'), 
                            30, 
                            (width,height))
    for image in images:
        # Process frames
        _, _, _, _, _, output_w_bboxes, output_img = process_image(image, cfg, net)
        # If --show-detections flag, use frames w/ bboxes
        if args.show_detections:
            video.write(output_w_bboxes)
        else:
            video.write(output_img)  
    # Release video writer object
    video.release()


def main(args):
    """Puts it all together."""
    # Start measuring time
    tic = time.perf_counter()
    # Load project configurations
    cfg = load_configs()
    # Load the network
    net = cv2.dnn.readNetFromTensorflow(cfg['net']['model_file'], 
                                        cfg['net']['cfg_file'])
    # Input and load image
    input_file = args.input

    try:
        # If file is a compatible video file
        if is_video(input_file):
            # Process video
            process_video(input_file, args.output, cfg, net)

        # If file is a compatible image file
        elif is_image(input_file):
            # Load image
            input_img = load_image(input_file)
            # Process image
            all_img_steps = process_image(input_img, cfg, net)
            # Save final image to specified output filename
            output_filename = os.path.join(args.output, cfg['image']['output'])
            if args.show_detections:
                output_img = all_img_steps[5]
            else:
                output_img = all_img_steps[6]
            img_saved = save_image(output_filename, output_img)

        # If input_file is a dir
        elif is_directory(input_file):
            # For each file in the dir
            for file in os.listdir(input_file):
                # Join input dir and file name
                file = os.path.join(input_file, file)
                # If file is a compatible video file
                if is_video(file):
                    # Process video
                    process_video(file, args.output, cfg, net)

                if is_image(file):
                    # Load image
                    input_img = load_image(file)
                    # Process image
                    all_img_steps = process_image(input_img, cfg, net)
                    # Save final image to specified output filename
                    output_filename = os.path.join(args.output, 
                                                   cfg['image']['output'])
                    if args.show_detections:
                        output_img = all_img_steps[5]
                    else:
                        output_img = all_img_steps[6]
                    img_saved = save_image(output_filename, output_img)

    except ValueError:
        print('Input must be a valid image, video, or directory.')
    
    # Save processing steps
    if args.save_steps:
        output_height = cfg['image']['img_steps_height']
        output_steps_filename = os.path.join(args.output, 
                                             cfg['image']['output_steps'])
        save_steps(output_steps_filename, all_img_steps, output_height)

    # End measuring time
    toc = time.perf_counter()
    print(f"Operation ran in {toc - tic:0.4f} seconds")


if __name__ == '__main__':
    args = parse_args()
    main(args)