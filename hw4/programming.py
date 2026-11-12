from typing import List, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------
# Challenge 1: Photometric Stereo
# --------------------------------------------------------------------------
def find_sphere(
    img: np.ndarray,
) -> Union[Tuple[Tuple[float, float], float], Tuple[None, None]]:
    """
    Locate the center of a (single) sphere in an image and compute its radius.

    Args:
        img (np.ndarray): Input image.

    Returns:
        Union[Tuple[Tuple[float, float], float], Tuple[None, None]]: The center and radius of the sphere, where the center
            is a 2 element tuple (x, y) representing the indices of the sphere center, and radius is the pixel radius of
            of the sphere. Return (None, None) if no sphere is located in the image.
    """
    gray = np.asarray(img, dtype=float)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    y, x = np.nonzero(gray > 0)
    if len(x) == 0:
        return None, None
    center = (float(x.mean()), float(y.mean()))
    radius = float(np.sqrt(len(x) / np.pi))
    return center, radius


def compute_light_directions(
    center: Tuple[float, float], radius: float, imgs: List[np.ndarray]
) -> np.ndarray:
    """
    Compute the direction to the light source.

    Args:
        center (Tuple[float, float]): Set of (x, y) coordinates to the sphere center.
        radius (float): Radius of the circle produced by the sphere on the image plane.
        imgs (List[np.ndarray]): An iterable of images of an object.

    Returns:
        np.ndarray: Nx3 array, where N=len(imgs), containing the normal vector to the brightest surface spot on
            the sphere in each image in imgs.
    """
    if radius <= 0:
        raise ValueError("The sphere radius must be positive")
    light_dirs = np.zeros((len(imgs), 3), dtype=float)
    for i, img in enumerate(imgs):
        gray = np.asarray(img, dtype=float)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
        yy, xx = np.indices(gray.shape)
        inside = (xx-center[0])**2 + (yy-center[1])**2 <= radius**2
        if not np.any(inside):
            raise ValueError("The sphere is outside the image")
        peak = gray[inside].max()
        if peak <= 0:
            continue
        y, x = np.nonzero(inside & (gray == peak))

        nx = (x.mean()-center[0]) / radius
        ny = (y.mean()-center[1]) / radius




        # The sphere satisfies (x-cx)^2+(y-cy)^2+z^2=R^2. Its outward
        # unit normal is (x-cx,y-cy,z)/R. With image x right and y down,
        # right-handed +z points into the scene, so the visible hemisphere
        # uses z=-sqrt(R^2-(x-cx)^2-(y-cy)^2).
        normal = np.array([nx, ny, -np.sqrt(max(0.0, 1-nx*nx-ny*ny))])
        normal /= np.linalg.norm(normal)





        # For a Lambertian sphere, brightness is proportional to
        # max(0,n dot s), which is greatest when the unit normal n equals
        # the light direction s. No specular component shifts this peak.
        # Scale that direction by the peak brightness to retain the
        # intensity of each source, rather than assuming equal intensities.
        light_dirs[i] = peak * normal
    return light_dirs


def compute_mask(imgs: List[np.ndarray]) -> np.ndarray:
    """
    Compute a binary foreground mask for the object.

    Args:
        imgs (List[np.ndarray]): An iterable of images of an object.

    Returns:
        np.ndarray: A binary mask where 1=object, 0=background.
    """
    images = np.asarray(imgs, dtype=float)
    if images.ndim == 4:
        images = images.mean(axis=3)
    threshold = 5.0
    if images.size and images.max() <= 1.0:
        threshold /= 255.0
    return (np.count_nonzero(images > threshold, axis=0) >= 3).astype(np.uint8)


def compute_normals(
    light_dirs: np.ndarray, imgs: List[np.ndarray], mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the normals and albedo to an object's surface.

    Args:
        light_dirs (np.ndarray): 5x3 lighting direction matrix from
            compute_light_directions.
        imgs (List[np.ndarray]): An iterable of images of an object.
        mask (np.ndarray): Binary foreground mask of an object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The surface normals and albedo of the object.
    """
    images = np.asarray(imgs, dtype=float)
    if images.ndim == 4:
        images = images.mean(axis=3)
    lights = np.asarray(light_dirs, dtype=float)
    foreground = np.asarray(mask, dtype=bool)
    height, width = foreground.shape
    normals = np.zeros((height, width, 3), dtype=float)
    normals[:, :, 2] = -1.0
    albedo_img = np.zeros((height, width), dtype=float)
    if not np.any(foreground):
        return normals, albedo_img
    intensities = images[:, foreground]
    threshold = 5.0 / 255.0 if images.max() <= 1.0 else 5.0
    selected = intensities > threshold
    insufficient = np.flatnonzero(selected.sum(axis=0) < 3)
    for column in insufficient:
        selected[np.argsort(intensities[:, column])[-3:], column] = True



    subsets = np.sum(selected * (1 << np.arange(len(imgs)))[:, None], axis=0)
    g = np.zeros((3, intensities.shape[1]), dtype=float)
    for subset in np.unique(subsets):
        rows = (int(subset) & (1 << np.arange(len(imgs)))) != 0
        columns = np.flatnonzero(subsets == subset)
        g[:, columns] = np.linalg.lstsq(lights[rows], intensities[rows][:, columns], rcond=None)[0]
    albedo = np.linalg.norm(g, axis=0)
    valid = albedo > np.finfo(float).eps
    foreground_normals = np.zeros((len(albedo), 3), dtype=float)
    foreground_normals[:, 2] = -1.0
    foreground_normals[valid] = (g[:, valid] / albedo[valid]).T
    normals[foreground] = foreground_normals
    maximum = albedo.max()
    if maximum > 0:
        albedo_img[foreground] = albedo / maximum
    return normals, albedo_img


def scene_relighting(
    mask: np.ndarray,
    normals: np.ndarray,
    albedo_img: np.ndarray,
    pointcloud: np.ndarray,
    point_source_xyz: np.ndarray,
    point_source_intensity: float,
) -> np.ndarray:
    """
    Compute the surface radiance at each point on the vase under the given
    illumination. Assume the vase is lambertian.

    The relit image will be scaled to [0, 1] using the lambertian shading
    equation. Before returning the image, you should scale it to a uint8 in
    the range [0, 255].

    Args:
        mask (np.ndarray): A binary mask where 1=object, 0=background.
        normals (np.ndarray): The normal vector at each point on the object image.
        albedo_img (np.ndarray): The albedo image of the object.
        pointcloud (np.ndarray): A the xyz location of each point in the image.
        point_source_xyz (np.ndarray): The xyz location of the point source illuminating the object.
        point_source_intensity (np.ndarray): The intensity location of the point source illuminating the object.

    Returns:
        np.ndarray: A relit image of the vase.
    """
    foreground = np.asarray(mask, dtype=bool)
    result = np.zeros(foreground.shape, dtype=np.uint8)
    if not np.any(foreground):
        return result
    displacement = np.asarray(point_source_xyz, dtype=float) - np.asarray(pointcloud, dtype=float)[foreground]
    distance_squared = np.sum(displacement * displacement, axis=1)
    valid = distance_squared > np.finfo(float).eps
    directions = np.zeros_like(displacement)
    directions[valid] = displacement[valid] / np.sqrt(distance_squared[valid, None])
    cosine = np.maximum(0.0, np.sum(np.asarray(normals)[foreground] * directions, axis=1))
    radiance = np.zeros(len(displacement), dtype=float)
    radiance[valid] = (np.asarray(albedo_img)[foreground][valid] / np.pi *
                       float(point_source_intensity) / distance_squared[valid] * cosine[valid])
    result[foreground] = np.rint(np.clip(radiance, 0.0, 1.0) * 255).astype(np.uint8)
    return result
