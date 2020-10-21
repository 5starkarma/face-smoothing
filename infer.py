import argparse
import yaml
import time

import cv2

import matplotlib
import matplotlib.pyplot as plt

from detector.detect import detect_face
from detector.smooth import smooth_face
from utils.image import load_image, save_image, save_steps


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


def main(args):
    tic = time.perf_counter()
    # Load project configurations
    cfg = load_configs()
    # Load the network
    net = cv2.dnn.readNetFromTensorflow(cfg['net']['model_file'], 
                                        cfg['net']['cfg_file'])
    # Input image
    input_file = args.input
    # Load image
    img = load_image(input_file)
    # Detect face
    box_face, bboxes = detect_face(net, input_file, 
                                   cfg['net']['conf_threshold'])
    toc = time.perf_counter()
    print(f"Face detected in {toc - tic:0.4f} seconds")
    # Smooth face and return steps
    box_face, roi_img, hsv_mask, full_img = smooth_face(cfg, box_face, bboxes)
    # Save final image with bbox
    img_saved = save_image(args.output, 
                           cfg['image']['output_with_bbox'], 
                           box_face)
    # Save processing steps
    if args.save_steps:
        save_steps(img, box_face, roi_img, hsv_mask, full_img)


if __name__ == '__main__':
    args = parse_args()
    main(args)