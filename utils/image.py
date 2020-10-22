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
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


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
    combined_img = np.concatenate(imgs, axis=1)
    return combined_img

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
    resized_imgs = tuple(resize_image(img, None, output_height) for img in all_img_steps)
    # Concatenate images horizontally
    combined_imgs = concat_imgs(resized_imgs)
    # Save concatenated image
    img_saved = save_image(filename, combined_imgs)
    return img_saved


