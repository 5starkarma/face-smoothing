import os


def is_image(file, ext=('.png', '.jpg', '.jpeg')):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def is_video(file, ext=('.mp4', '.avi', '.mov', '.mpeg', '.flv', '.wmv')):
    """Check if a file is a video with certain extensions"""
    return file.endswith(ext)

def is_directory(file):
    """Check if input is a directory"""
    return os.path.isdir(file)

