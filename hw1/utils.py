import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def print_banner(s: str = ""):
    """
    Print a banner to stdout.

    Args:
        s (str): Banner heading.
    """
    if s:
        print(s)
    try:
        banner_width = int(os.popen("stty size", "r").read().split()[-1])
    except IndexError:
        banner_width = 30
    print("=" * banner_width)


def get_data_path(filename: str) -> str:
    """
    Return the path to a data file.

    Args:
        filename (str): Name of the file.

    Returns:
        str: The absolute path to the file.
    """
    return str((DATA_DIR / filename).resolve())


def get_result_path(filename: str) -> str:
    """
    Return the path to a data file.

    Args:
        filename (str): Name of the file.

    Returns:
        str: The absolute path to the file.
    """
    return str((RESULTS_DIR / filename).resolve())


def imread(
    path: str, flag: int = cv2.IMREAD_COLOR, rgb: bool = False, normalize: bool = False
) -> np.ndarray:
    """
    Read an image from a file.

    Args:
        path (str): Path to the image.
        flag (int, optional): Image read flag passed to cv2.imread. Defaults to cv2.IMREAD_COLOR.
        rgb (bool, optional): Whether to read the image as BGR (rgb=False) or RGB (rgb=True).
            Defaults to False.
        normalize (bool, optional): Normalize the image to the range [0, 1]. Defaults to False.

    Raises:
        FileNotFoundError: If the filepath could not be found.

    Returns:
        np.ndarray: The image as a multidimensional array.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(str(path), flag)

    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float32) / 255
    return img


def imread_alpha(path: str, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an image from a file.
    Use this function when the image contains an alpha channel. That channel
    is returned separately.

    Args:
        path (str): Path to the image.
        normalize (bool, optional): Normalize the image to the range [0, 1]. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the image and alpha channel.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if normalize:
        img = img.astype(np.float32) / 255

    alpha = img[:, :, -1]
    img = img[:, :, :-1]

    return img, alpha


def imshow(img: np.ndarray, title: str = None, flag: int = cv2.COLOR_BGR2RGB):
    """
    Display an image in a windowed viewer.

    Args:
        img (np.ndarray): Input image.
        title (str, optional): Title of the window. Defaults to None.
        flag (int, optional): cv2.cvtColor flag. Defaults to cv2.COLOR_BGR2RGB.
    """
    plt.figure()
    if flag is not None:
        if img.dtype == np.float64:
            img = img.astype(np.float32)
        img = cv2.cvtColor(img, flag)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def imwrite(path: str, img: np.ndarray, flag: Optional[int] = None):
    """
    Write an image to a file.

    Args:
        path (str): Path to save the image to.
        img (np.ndarray): The input image.
        flag (Optional[int], optional): cv2.cvtColor flag. Defaults to None.
    """
    assert type(img) == np.ndarray
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if flag is not None:
        img = cv2.cvtColor(img, flag)
    cv2.imwrite(str(path), img)
