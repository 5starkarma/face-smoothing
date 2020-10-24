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
                         check_img_size)
from utils.video import split_video
from utils.types import (is_image,
                         is_video)


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


def process_image(input_img, cfg, net):
    # Make sure image is less than 1081px wide
    input_img = check_img_size(input_img)
    # Detect face
    detected_img, bboxes = detect_face(cfg,
                                       net, 
                                       input_img, 
                                       cfg['net']['conf_threshold'])
    # Smooth face and return steps
    output_img, roi_img, hsv_mask, smoothed_roi = smooth_face(cfg, input_img, bboxes)
    return (input_img, detected_img, roi_img, hsv_mask, smoothed_roi, output_img)


def process_video(file, output_dir, cfg, net):
    # Split video into frames
    images = split_video(file)
    counter = 0
    # Add brackets and extension to filename
    filename = os.path.join(output_dir, cfg['video']['output']) + '{}.mp4'
    # If a file of this name exists increase the counter by 1
    while os.path.isfile(filename.format(counter)):
        counter += 1
    # Apply counter to filename
    output_path = filename.format(counter)
    # Process images in folder
    processed_images = []
    # Get height and width of 1st image
    input_img = check_img_size(images[0])
    height, width, _ = input_img.shape  
    print(height, width) 
    # Create VideoWriter object
    video = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'FMP4'), 
                            30, 
                            (width,height))
    for image in images:
        # Process image
        _, detected_img, _, _, _, output_img = process_image(image, cfg, net)
        # Write output images to video 
        video.write(output_img)  
        # Release video writer object
    video.release()  
    # Delete input video
    # delete_video(file)


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
    # If file is a compatible video file
    if is_video(input_file):
        # Process video
        process_video(input_file, args.output, cfg, net)
        # Merge all files in folder back to video

    # If file is a compatible image file
    elif is_image(input_file):
        input_img = load_image(input_file)
        all_img_steps = process_image(input_img, cfg, net)
        # Save final image without bbox
        output_filename = os.path.join(args.output, cfg['image']['output'])
        img_saved = save_image(output_filename, all_img_steps[5])

    # If input_file is a dir
    elif os.path.isdir(input_file):
        # For each file in the dir
        for file in os.listdir(input_file):
            # Join input dir and file name
            file = os.path.join(input_file, file)
            # If file is a compatible video file
            if is_video(file):
                # Process video
                process_video(file, args.output, cfg, net)
                # Merge all files in folder back to video

            if is_image(file):
                input_img = load_image(file)
                all_img_steps = process_image(input_img, cfg, net)
                output_filename = os.path.join(args.output, cfg['image']['output'])
                img_saved = save_image(output_filename, all_img_steps[5])
            
            # While a file in dir or subdir is compatible image:
            
                # process images
    else: 
        print(f'Unable to process files from: {input_file}')

    output_height = cfg['image']['img_steps_height']
    
    # Save processing steps
    if args.save_steps:
        output_steps_filename = os.path.join(args.output, cfg['image']['output_steps'])
        save_steps(output_steps_filename, all_img_steps, output_height)

    # End measuring time
    toc = time.perf_counter()
    print(f"Operation ran in {toc - tic:0.4f} seconds")


if __name__ == '__main__':
    args = parse_args()
    main(args)