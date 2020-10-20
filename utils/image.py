# Save image adapted from https://stackoverflow.com/a/56680778/10796680

# System imports
import os

#Third-party imports
import cv2


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
    filename : str
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