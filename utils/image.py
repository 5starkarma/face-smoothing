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