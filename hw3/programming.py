from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import utils
from scipy.ndimage import distance_transform_edt

# Tunable parameters

# RANSAC parameters for challenge1a
CHALLENGE_1C_RANSAC_N = 500  # Number of iterations
CHALLENGE_1C_RANSAC_EPS = 5  # Maximum reprojection error


# --------------------------------------------------------------------------
# Challenge 1: Image Mosaicking App
# --------------------------------------------------------------------------
def compute_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    """
    Compute the homography matrix relating the given points.
    Hint: use np.linalg.eig to compute the eigenvalues of a matrix.

    Args:
        src_pts (np.ndarray): Nx2 matrix of source points
        dest_pts (np.ndarray): Nx2 matrix of destination points

    Returns:
        np.ndarray: 3x3 homography matrix
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2
    N = src_pts.shape[0]

    if N < 4:
        raise ValueError("At least four correspondences are required")
    normalized = []
    transforms = []
    for points in (src_pts, dest_pts):
        points = np.asarray(points, dtype=float)
        center = points.mean(axis=0)
        mean_distance = np.linalg.norm(points-center, axis=1).mean()
        if mean_distance < 1e-12:
            raise ValueError("Degenerate point configuration")
        scale = np.sqrt(2) / mean_distance
        T = np.array([[scale, 0, -scale*center[0]],
                      [0, scale, -scale*center[1]], [0, 0, 1]])
        normalized.append((points-center)*scale)
        transforms.append(T)
    x, y = normalized[0].T
    u, v = normalized[1].T
    A = np.zeros((2*N, 9))
    A[0::2] = np.column_stack((-x, -y, -np.ones(N), np.zeros((N, 3)), u*x, u*y, u))
    A[1::2] = np.column_stack((np.zeros((N, 3)), -x, -y, -np.ones(N), v*x, v*y, v))
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("Point configuration does not determine a homography")
    eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
    H_normalized = eigenvectors[:, np.argmin(eigenvalues.real)].real.reshape(3, 3)
    H = np.linalg.inv(transforms[1]) @ H_normalized @ transforms[0]
    scale = H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
    return H / scale


def apply_homography(H: np.ndarray, test_pts: np.ndarray) -> np.ndarray:
    """
    Apply the homography to the given test points

    Args:
        H (np.ndarray): 3x3 homography matrix
        test_pts (np.ndarray): Nx2 test points

    Returns:
        np.ndarray: Nx2 points after applying the homography
    """
    assert test_pts.shape[1] == 2

    points = np.column_stack((np.asarray(test_pts, dtype=float), np.ones(len(test_pts))))
    transformed = points @ np.asarray(H, dtype=float).T
    result = np.full((len(test_pts), 2), np.nan)
    valid = np.abs(transformed[:, 2]) > 1e-12
    result[valid] = transformed[valid, :2] / transformed[valid, 2, None]
    return result


def backward_warp_img(
    src_img: np.ndarray, H_inv: np.ndarray, dest_canvas_width_height: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a homography to the image using backward warping.

    Use cv2.remap to interpolate the warped points.
    The function call should follow this form:
        img_warp = cv2.remap(img, map_x.astype(np.float32),
            map_y.astype(np.float32), cv2.INTER_LINEAR, borderValue=np.nan).
    This casts map_x and map_y to 32-bit floats, chooses linear interpolation,
    and sets pixels outside the original image to NaN (not-a-number).

    Also, since we are working with color images, you should process each
    color channel separately.

    Args:
        src_img (np.ndarray): Nx2 source points
        H_inv (np.ndarray): 3x3 inverse of the src -> dest homography
        dest_canvas_width_height (Tuple[int, int]): size of the destination image

    Returns:
        Tuple[np.ndarray, np.ndarray]: final image, binary mask where the
            destination image is filled in
    """

    width, height = dest_canvas_width_height
    y, x = np.indices((height, width))
    points = np.column_stack((x.ravel(), y.ravel()))
    source_points = apply_homography(H_inv, points)
    map_x = source_points[:, 0].reshape(height, width)
    map_y = source_points[:, 1].reshape(height, width)
    source_height, source_width = src_img.shape[:2]
    valid = (np.isfinite(map_x) & np.isfinite(map_y) &
             (map_x >= -1e-6) & (map_x <= source_width-1+1e-6) &
             (map_y >= -1e-6) & (map_y <= source_height-1+1e-6))
    map_x = np.where(valid, np.clip(map_x, 0, source_width-1), -100)
    map_y = np.where(valid, np.clip(map_y, 0, source_height-1), -100)
    source = np.asarray(src_img, dtype=float)
    channels = source[:, :, None] if source.ndim == 2 else source
    warped = np.empty((height, width, channels.shape[2]), dtype=float)
    for channel in range(channels.shape[2]):
        warped[:, :, channel] = cv2.remap(
            np.pad(channels[:, :, channel], 1, mode="edge"),
            (map_x+1).astype(np.float32), (map_y+1).astype(np.float32),
            cv2.INTER_LINEAR, borderValue=np.nan)
    mask = valid & np.all(np.isfinite(warped), axis=2)
    warped[~mask] = 0
    return (warped[:, :, 0] if source.ndim == 2 else warped), mask


def warp_img_onto(
    src_img: np.ndarray, dest_img: np.ndarray, src_pts: np.ndarray, dest_pts: np.ndarray
) -> np.ndarray:
    """
    Warp the source image on the destination image.
    Return the resulting image.

    Step 1: estimate the homography mapping src_pts to dest_pts
    Step 2: warp src_img onto dest_img using backward_warp_img(..)
    Step 3: overlay the warped src_img onto dest_img
    Step 4: return the resulting image

    Args:
        src_img (np.ndarray): source image
        dest_img (np.ndarray): destination image
        src_pts (np.ndarray): Nx2 source points
        dest_pts (np.ndarray): Nx2 destination points

    Returns:
        np.ndarray: resulting image with the source image warped on the
        destination image
    """
    H = compute_homography(src_pts, dest_pts)
    height, width = dest_img.shape[:2]
    warped, mask = backward_warp_img(src_img, np.linalg.inv(H), (width, height))
    return blend_image_pair(dest_img, np.ones((height, width), dtype=bool),
                            warped, mask, "overlay")


def run_RANSAC(
    src_pts: np.ndarray, dest_pts: np.ndarray, ransac_n: int, ransac_eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run RANSAC on the given point correspondences to compute a homography
    from source to destination points. When sampling new points at each RANSAC
    iteration, remember to sample _without_ replacement.

    Args:
        src_pts (np.ndarray): Nx2 source points
        dest_pts (np.ndarray): Nx2 destination points
        ransac_n (int): number of RANSAC iterations
        ransac_eps (float): maximum 2D reprojection error for inliers

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the inlier indices and the
        estimated homography matrix.
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2

    if len(src_pts) < 4 or ransac_n <= 0 or ransac_eps <= 0:
        raise ValueError("RANSAC needs four matches and positive parameters")
    rng = np.random.default_rng(0)
    best_indices = np.empty(0, dtype=int)
    best_error = np.inf
    best_H = None
    for _ in range(ransac_n):
        sample = rng.choice(len(src_pts), 4, replace=False)
        try:
            H = compute_homography(src_pts[sample], dest_pts[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        error = np.linalg.norm(apply_homography(H, src_pts)-dest_pts, axis=1)
        indices = np.flatnonzero(np.isfinite(error) & (error <= ransac_eps))
        mean_error = error[indices].mean() if len(indices) else np.inf
        if len(indices) > len(best_indices) or (len(indices) == len(best_indices) and mean_error < best_error):
            best_indices, best_error, best_H = indices, mean_error, H
    if best_H is None or len(best_indices) < 4:
        raise ValueError("Could not find a nondegenerate consensus")
    for _ in range(5):
        try:
            refined_H = compute_homography(src_pts[best_indices], dest_pts[best_indices])
        except (ValueError, np.linalg.LinAlgError):
            break
        error = np.linalg.norm(apply_homography(refined_H, src_pts)-dest_pts, axis=1)
        indices = np.flatnonzero(np.isfinite(error) & (error <= ransac_eps))
        if len(indices) < 4:
            break
        unchanged = np.array_equal(indices, best_indices)
        best_indices, best_H = indices, refined_H
        if unchanged:
            break
    return best_indices, best_H


def blend_image_pair(
    img1: np.ndarray,
    mask1: np.ndarray,
    img2: np.ndarray,
    mask2: np.ndarray,
    blending_mode: str,
) -> np.ndarray:
    """
    Blend two images together using the "overlay" or "blend" mode.

    In the "overlay" mode, image 2 is overlaid on top of image 1 wherever
    mask2 is non-zero.
    In "blend" mode, the blended image is a weighted combination of the two
    images, where each pixel is weighted based on its distance to the edge.

    Args:
        img1 (np.ndarray): First image
        mask1 (np.ndarray): Mask where the first image is non-zero
        img2 (np.ndarray): Second image
        mask2 (np.ndarray): Mask where the second image is non-zero
        blending_mode (str): "overlay" or "blend"

    Returns:
        np.ndarray: blended image.
    """
    assert blending_mode in ["overlay", "blend"]

    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    first = np.where(mask1[..., None], np.nan_to_num(img1), 0) if img1.ndim == 3 else np.where(mask1, np.nan_to_num(img1), 0)
    second = np.where(mask2[..., None], np.nan_to_num(img2), 0) if img2.ndim == 3 else np.where(mask2, np.nan_to_num(img2), 0)
    if blending_mode == "overlay":
        return np.where(mask2[..., None] if img2.ndim == 3 else mask2, second, first)
    weight1 = distance_transform_edt(np.pad(mask1, 1))[1:-1, 1:-1]
    weight2 = distance_transform_edt(np.pad(mask2, 1))[1:-1, 1:-1]
    total = weight1 + weight2
    total[total == 0] = 1
    if img1.ndim == 3:
        weight1, weight2, total = weight1[..., None], weight2[..., None], total[..., None]
    return (first*weight1 + second*weight2) / total


def stitch_imgs(imgs: List[np.ndarray]) -> np.ndarray:
    """
    Stitch a list of images together into an image mosaic.
    imgs: list of images to be stitched together. You may assume the order
    the images appear in the list is the order in which they should be stitched
    together.

    Args:
        imgs (List[np.ndarray]): list of images to be stitched together. You may
        assume the order the images appear in the list is the order in which
        they should be stitched together.

    Returns:
        np.ndarray: the final, stitched image
    """

    if not imgs:
        raise ValueError("At least one image is required")
    imgs = [image.astype(float) / np.iinfo(image.dtype).max
            if np.issubdtype(image.dtype, np.integer) else np.asarray(image, dtype=float)
            for image in imgs]
    mosaic = imgs[0].copy()
    mosaic_mask = np.ones(mosaic.shape[:2], dtype=bool)
    for image in imgs[1:]:
        src_pts, dest_pts = utils.sift_matches(image, mosaic)
        _, H = run_RANSAC(src_pts, dest_pts, CHALLENGE_1C_RANSAC_N, CHALLENGE_1C_RANSAC_EPS)
        height, width = image.shape[:2]
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        transformed_corners = apply_homography(H, corners)
        if not np.isfinite(transformed_corners).all():
            raise ValueError("Image corners project to infinity")
        old_height, old_width = mosaic.shape[:2]
        minimum = np.floor(np.minimum(transformed_corners.min(axis=0), [0, 0])).astype(int)
        maximum = np.ceil(np.maximum(transformed_corners.max(axis=0), [old_width-1, old_height-1])).astype(int)
        new_width, new_height = maximum-minimum+1
        if new_width*new_height > 50_000_000:
            raise ValueError("Implausibly large mosaic; check image overlap")
        T = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]])
        padded = np.zeros((new_height, new_width)+mosaic.shape[2:], dtype=float)
        padded_mask = np.zeros((new_height, new_width), dtype=bool)
        left, top = -minimum
        padded[top:top+old_height, left:left+old_width] = mosaic
        padded_mask[top:top+old_height, left:left+old_width] = mosaic_mask
        warped, mask = backward_warp_img(image, np.linalg.inv(T @ H), (new_width, new_height))
        mosaic = blend_image_pair(padded, padded_mask, warped, mask, "blend")
        mosaic_mask = padded_mask | mask
    return mosaic


def build_your_own_panorama() -> np.ndarray:
    """
    Build your own panorama using images in results/challenge_1f_input/.
    """
    input_path = utils.get_result_path("challenge_1f_input")  # Do not change

    # Load images
    # Start with the middle view to limit perspective distortion.
    file_names = sorted(p.name for p in Path(input_path).iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}) if Path(input_path).is_dir() else []
    middle = len(file_names) // 2
    file_names = file_names[middle:middle+1] + file_names[:middle][::-1] + file_names[middle+1:]
    imgs = []
    for f_name in file_names:
        img_path = str((Path(input_path) / f_name).resolve())
        img = utils.imread(img_path, flag=None, rgb=True, normalize=True)
        height, width = img.shape[:2]
        scale = min(1.0, 1280/width, 720/height)
        if scale < 1:
            img = cv2.resize(img, (round(width*scale), round(height*scale)))
        imgs.append(img)

    if len(imgs) < 2:
        raise FileNotFoundError("Add at least two overlapping camera photos to results/challenge_1f_input")
    return stitch_imgs(imgs)
