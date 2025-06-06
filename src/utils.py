import os
import cv2 as cv
import numpy as np
from typing import Tuple


def get_paths(base_path: str) -> Tuple[dict, dict, dict]:
    """
    Returns the paths to the training, testing, and validation datasets.

    The base_path should contain three directories: train, test, and val.
    Each of these directories should contain two subdirectories for each label

    Parameters
    ----------
    base_path : str
        The base path where the dataset directories are located.
    Returns
    -------
    train : dict
        A dictionary containing paths for training data, with keys for normal
        and abnormal labels.
    test : dict
        A dictionary containing paths for testing data, with keys for normal
        and abnormal labels.
    val : dict
        A dictionary containing paths for validation data, with keys for normal
        and abnormal labels.
    """

    paths = {
        "test": {"NORMAL": "", "PNEUMONIA": ""},
        "train": {"NORMAL": "", "PNEUMONIA": ""},
        "val": {"NORMAL": "", "PNEUMONIA": ""}
    }

    for dir in os.listdir(base_path):

        almost_full_path = os.path.join(base_path, dir)

        for label in os.listdir(almost_full_path):

            full_path = os.path.join(almost_full_path, label)

            if dir == "test":
                if label.upper() == "NORMAL":
                    paths["test"]["NORMAL"] = full_path
                else:
                    paths["test"]["PNEUMONIA"] = full_path
            elif dir == "train":
                if label.upper() == "NORMAL":
                    paths["train"]["NORMAL"] = full_path
                else:
                    paths["train"]["PNEUMONIA"] = full_path
            elif dir == "val":
                if label.upper() == "NORMAL":
                    paths["val"]["NORMAL"] = full_path
                else:
                    paths["val"]["PNEUMONIA"] = full_path

    return paths


def read_image(image_path: str) -> np.ndarray:
    """
    Reads an image in grayscale mode using OpenCV.

    Parameters
    ----------
    image_path : str
        Full path to the image file.

    Returns
    -------
    image : np.ndarray
        Grayscale image as a 2D NumPy array.
    """
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    return image