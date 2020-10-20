# Save image adapted from https://stackoverflow.com/a/56680778/10796680

# System imports
import os

#Third-party imports
import cv2
import numpy as np


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


def save_image(output_dir, filename, img):
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
    filename = os.path.join(output_dir, filename + '{}.jpg')
    while os.path.isfile(filename.format(counter)):
        counter += 1
    filename = filename.format(counter)
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

    
def resize_image(image, width=None, height=None):
    """source: *1* at bottom"""
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
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def save_steps(img, box_face, roi_img, hsv_mask, full_img):
    img = resize_image(img, height=300)
    box_face = resize_image(box_face, height=300)
    roi_img = resize_image(roi_img, height=300)
    hsv_mask = resize_image(hsv_mask, height=300)
    full_img = resize_image(full_img, height=300)
    combined_imgs = np.concatenate((img, box_face, roi_img, hsv_mask, full_img), axis=1)
    cv2.imwrite('/content/drive/My Drive/Colab Notebooks/face-smoothing/data/output/combined.jpg', combined_imgs)