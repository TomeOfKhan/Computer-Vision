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


def reconstruct_surf(normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Surface reconstruction using the Frankot-Chellappa algorithm

    Args:
        normals (np.ndarray): An array of normal vectors on an object.
        mask (np.ndarray): The foreground mask for an object.

    Returns:
        np.ndarray: The surface image.
    """
    # Compute surface gradients (p, q)
    p_img = normals[:, :, 0] / -(normals[:, :, 2] + np.finfo(float).eps)
    q_img = normals[:, :, 1] / -(normals[:, :, 2] + np.finfo(float).eps)

    # Take Fourier Transform of p and q
    fp_img = np.fft.fft2(p_img)
    fq_img = np.fft.fft2(q_img)
    (cols, rows) = fp_img.shape

    # The domains of u and v are important
    (u, v) = np.meshgrid(
        np.arange(cols) - np.fix(cols / 2), np.arange(rows) - np.fix(rows / 2)
    )
    u = np.fft.ifftshift(u)
    v = np.fft.ifftshift(v)
    fz = (1j * u * fp_img + 1j * v * fq_img) / (u**2 + v**2 + np.finfo(float).eps)

    # Take inverse Fourier Transform back to the spatial domain
    ifz = np.fft.ifft2(fz)
    ifz[~mask] = 0
    z = np.real(ifz)
    surf_img = (z - np.min(z)) / (np.max(z) - np.min(z))
    surf_img[~mask] = 0

    return surf_img


def euler_to_rotm(theta: np.ndarray) -> np.ndarray:
    """
    Calculate the rotation matrix given a set of Euler angles.

    Args:
        theta (np.ndarray): The Euler angles (euler_x, euler_y, euler_z) in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    Ry = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = Rz @ (Ry @ Rx)
    return R


class Orbit:
    # Define the point source orbit
    def __init__(
        self,
        start_position,  # unit distance (meters)
        euler_step=np.deg2rad([0, 0, 2]),  # radians
        translation_step=np.array([0, 0, 0]),  # unit distance (meters)
    ) -> None:
        # Preliminaries
        self.start_position = start_position
        self.euler_step = euler_step
        self.translation_step = translation_step

        # Cache
        self.xyz = self.start_position.copy()

    def step(self, dt: int = 1) -> np.ndarray:
        """
        Step the orbiter to get the next set of xyz coordinates.

        Args:
            dt (int, optional): The length of the timestep to take. Defaults to 1.

        Returns:
            np.ndarray: The next set of xyz coordinates for the orbiter.
        """
        R = euler_to_rotm(dt * self.euler_step)
        T = dt * self.translation_step
        self.xyz = R @ self.xyz + T
        return self.xyz.copy()
