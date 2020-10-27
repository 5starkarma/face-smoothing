# Save image adapted from https://stackoverflow.com/a/56680778/10796680

# System imports
import os

# Third-party imports
import cv2
import numpy as np

from detector import detect, smooth


def resize_image(image, width=None, height=None):
    """
    Resize image with proportionate scaling. e.g. If 
    only width is given, height will automatically 
    proportionally scale.

    Source
    ------
    https://stackoverflow.com/a/56859311/10796680

    Parameters
    ----------
    img : np.array [H, W, 3]
        RGB image

    Returns
    -------
    image shape : int, int
        height and width of image
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, 
                         dim, 
                         interpolation=cv2.INTER_AREA)
    return resized


def check_img_size(img):
    """
    Verifies that the image is 360x540 or smaller
    to help the detector find faces.
    """
    # Retrieve image size
    height, width = img.shape[:2]
    # If image h is > 720 or w is > 1080, resize
    if height > 720 or width > 1080:
        img = resize_image(img, 
                           width=720 if width > 720 else None, 
                           height=1080 if height > 1080 else None)
    return img


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
    detected_img, bboxes = detect.detect_face(cfg, net, input_img)
    # Smooth face and return steps
    output_img, roi_img, hsv_mask, smoothed_roi = smooth.smooth_face(cfg, 
                                                              input_img, 
                                                              bboxes)
    # Draw bboxes on output_img
    output_w_bboxes = draw_bboxes(output_img, cfg, bboxes)
    return (input_img, detected_img, roi_img, hsv_mask, 
            smoothed_roi, output_w_bboxes, output_img)


def load_image(path):
    """
    Read an image using OpenCV

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------
    image : np.array [H,W,3]
        RGB image
    """
    return cv2.imread(path)


def save_image(filename, img):
    """
    Save an image using OpenCV

    Parameters
    ----------
    output_dir : str
        Name to save image as
    filename : str
        Name to save image as
    img : str
        Name to save image as

    Returns
    -------
    Bool : bool
        True if image save was success
    """
    counter = 0
    # Add brackets and extension to filename
    filename = filename + '{}.jpg'
    # If a file of this name exists increase the counter by 1
    while os.path.isfile(filename.format(counter)):
        counter += 1
    # Apply counter to filename
    filename = filename.format(counter)
    # Save file
    return cv2.imwrite(filename, img)


def get_height_and_width(img):
    """
    Retrieve height and width of image

    Parameters
    ----------
    img : np.array [H, W, 3]
        RGB image

    Returns
    -------
    image shape : int, int
        height and width of image
    """
    return img.shape[0], img.shape[1]


def concat_imgs(imgs):
    """
    Concatenates tuple of images.

    Parameters
    ----------
    imgs : tuple
        tuple of BGR images

    Returns
    -------
    combined_img : BGR image
        Image of horizontally stacked images
    """
    # Horizontally concatenate images
    return np.concatenate(imgs, axis=1)

def save_steps(filename, all_img_steps, output_height):
    """
    Resizes and concatenates tuple of images.

    Parameters
    ----------
    filename : str
        Output filename

    all_img_steps : tuple
        Tuple of BGR images

    output_height : int
        Height of output image

    Returns
    -------
    img_saved : bool
        True if successful save
    """
    # Map resized images
    resized_imgs = tuple(resize_image(img, None, output_height) 
                                      for img in all_img_steps)
    # Concatenate images horizontally
    combined_imgs = concat_imgs(resized_imgs)
    # Save concatenated image
    return save_image(filename, combined_imgs)


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


def check_if_adding_bboxes(args, img_steps):
    """
    Check if --show-detections flag is given. 
    If it is, return the image with bboxes.

    Parameters
    ----------
    args : Namespace object
        ArgumentParser

    img_steps : tuple
        Tuple of image steps

    Returns
    -------
    configs : dict
        A dictionary containing the configs
    """
    # If --show-detections flag show image w/ bboxes
    if args.show_detections:
        return img_steps[5]
    else:
        return img_steps[6]


