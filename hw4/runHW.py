# If any of packages fail to import, it is like that you have not correctly
# installed the course anaconda environment.
import argparse
import io
from typing import Callable, List, Tuple

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import utils
from PIL import Image

try:
    import solutions as programming
except ImportError:
    import programming


def runHW(fns: List[Tuple[str, Callable]]):
    """
    runHW is the "main" interface that lets you execute all the
    walkthroughs and challenges in this homework. It lists a set of
    functions corresponding to the problems that need to be solved.

    This file also serves as specifications for the functions
    you are asked to implement. In some cases, your submissions will be
    autograded.  Thus, it is critical that you adhere to all the specified
    function signatures.

    Before your submssion, make sure you can run all functions without any
    error.

    Usage:
    python runHW -h                               : show help and list all the registered functions
    python runHW --run_fn honesty                 : execute a specific function
    python runHW --run_fn honesty walkthrough1    : execute a set of specific functions
    python runHW --run_all                        : execute all the registered functions
    """
    np.set_printoptions(precision=4, suppress=True)

    for fn_name, fn_callable in fns:
        utils.print_banner(fn_name)
        fn_callable()
        utils.print_banner()
        print()


# --------------------------------------------------------------------------
# Tests for Challenge 1: Photometric Stereo
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Tests for Challenge 1a: Find Sphere
# --------------------------------------------------------------------------
def challenge1a():
    # Compute the properties of the sphere
    img = utils.imread(utils.get_data_path("sphere0.png"), cv2.IMREAD_GRAYSCALE)
    utils.imshow(img, title="sphere0")

    center, radius = programming.find_sphere(img)
    print(f"sphere {center = }")
    print(f"sphere {radius = }")

    np.savez(
        utils.get_result_path("sphere_properties.npz"),
        center=center,
        radius=radius,
    )


# --------------------------------------------------------------------------
# Tests for Challenge 1b: Compute Light Directions
# --------------------------------------------------------------------------
def challenge1b():
    # Compute the directions of light sources
    imgs = []
    for i in range(1, 6):
        img = utils.imread(
            utils.get_data_path(f"sphere{i}.png"), cv2.IMREAD_GRAYSCALE
        ).astype(np.uint8)
        imgs.append(img)

    data = np.load(utils.get_result_path("sphere_properties.npz"))
    center, radius = data["center"], data["radius"]

    light_dirs_5x3 = programming.compute_light_directions(center, radius, imgs)
    print(f"sphere light_dirs_5x3 = \n{light_dirs_5x3}")

    np.savez(utils.get_result_path("light_dirs.npz"), light_dirs_5x3=light_dirs_5x3)


# --------------------------------------------------------------------------
# Tests for Challenge 1c: Compute Mask
# --------------------------------------------------------------------------
def challenge1c():
    # Compute the mask of the object
    vase_imgs = []
    for i in range(1, 6):
        img = utils.imread(utils.get_data_path(f"vase{i}.png")).astype(np.uint8)
        vase_imgs.append(img)

    mask = programming.compute_mask(vase_imgs)
    mask = (mask * 255).astype(np.uint8)
    utils.imshow(mask, title="vase mask")

    utils.imwrite(utils.get_result_path("vase_mask.png"), mask)


# --------------------------------------------------------------------------
# Tests for Challenge 1d: Compute Normals
# --------------------------------------------------------------------------
def challenge1d():
    # Compute surface normals and albedos of the object
    mask = utils.imread(
        utils.get_result_path("vase_mask.png"), cv2.IMREAD_GRAYSCALE
    ).astype(bool)
    data = np.load(utils.get_result_path("light_dirs.npz"))
    light_dirs_5x3 = data["light_dirs_5x3"]

    vase_imgs = []
    for i in range(1, 6):
        img = utils.imread(
            utils.get_data_path(f"vase{i}.png"), cv2.IMREAD_GRAYSCALE
        ).astype(np.uint8)
        vase_imgs.append(img)

    normals, albedo_img = programming.compute_normals(light_dirs_5x3, vase_imgs, mask)
    normal_map_img = ((normals + 1) / 2 * 255).astype(np.uint8)
    albedo_img = (albedo_img * 255).astype(np.uint8)

    utils.imshow(normal_map_img, title="vase normal map")
    utils.imshow(albedo_img, title="vase albedo")

    utils.imwrite(utils.get_result_path("vase_normal_map.png"), normal_map_img)
    utils.imwrite(utils.get_result_path("vase_albedo.png"), albedo_img)
    np.savez(utils.get_result_path("normals.npz"), normals=normals)


# --------------------------------------------------------------------------
# Demonstrate Surface Reconstruction
# --------------------------------------------------------------------------
def demo_surface_reconstruction():
    # Demo (no submission required)
    data = np.load(utils.get_result_path("normals.npz"))
    normals = data["normals"]
    mask = utils.imread(
        utils.get_result_path("vase_mask.png"), cv2.IMREAD_GRAYSCALE
    ).astype(bool)
    surf_img = utils.reconstruct_surf(normals, mask)

    # Resize the surf_img
    resized_surf_img = cv2.resize(surf_img, None, fx=0.3, fy=0.3)

    # Create a meshgrid for the x and y coordinates
    x = np.arange(resized_surf_img.shape[1])
    y = np.arange(resized_surf_img.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface with color based on depth
    surf = ax.plot_surface(X, Y, resized_surf_img, cmap="viridis")

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()


# --------------------------------------------------------------------------
# Tests for Challenge 1e: Relight the scene under an orbiting light source
# --------------------------------------------------------------------------
INTENSITY = 255e4  # point source intensity
NUM_ORBITS = 2  # number of point source orbits
ORBIT_RADIUS_FACTOR = (
    3  # multiples of image height and width with which to set orbit radius
)
POINT_SOURCE_VIS_SIZE = 100  # size of the point source in visualization
DELTA_T = 4  # number of orbit steps to take on each iteration of visualization
GAMMA = 1.0  # gamma correction for visualization
Z_SCALE_FACTOR = 3  # scaling to make object look nicer
Z_OFFSET_PX = 250  # "height" of the orbit above the x-y axis
IMG2LEC_COORDINATE_SYSTEM = [
    1,
    -1,
    -1,
]  # reflect over x-z plane, reflect over x-z plane to match lecture coordinate system

gamma_correction = (
    lambda I, gamma, I_max: (I / I.max(axis=(0, 1))) ** (1 / gamma) * I_max
)


def challenge1e():
    # Load data from previous sections
    mask = utils.imread(
        utils.get_result_path("vase_mask.png"), cv2.IMREAD_GRAYSCALE
    ).astype(bool)
    normals = np.load(utils.get_result_path("normals.npz"))["normals"]
    albedo_img = utils.imread(
        utils.get_result_path("vase_albedo.png"), cv2.IMREAD_GRAYSCALE, normalize=True
    )
    surf_img = utils.reconstruct_surf(normals, mask)

    # Convert surface reconstruction into a pointcloud
    x = np.arange(mask.shape[1])
    y = np.arange(mask.shape[0])
    X, Y = np.meshgrid(x, y)
    max_xy = np.max((X, Y))
    Z = surf_img.copy() * max_xy / Z_SCALE_FACTOR  # scaling to make it look nicer
    points = (
        np.stack((X, Y, Z), axis=-1) * IMG2LEC_COORDINATE_SYSTEM
    )  # reflect over x-z plane, reflect over x-z plane

    # Define the orbit paths
    h, w = mask.shape[:2]
    points -= points.mean(axis=(0, 1))  # center vase at the origin
    orbit_start_pos = ORBIT_RADIUS_FACTOR * np.array([w, h, Z_OFFSET_PX])
    orbit = utils.Orbit(start_position=orbit_start_pos)

    # Initialize the visualization
    light_source_xyz = orbit.step(dt=0)
    vis_axis_range = np.linalg.norm(orbit_start_pos[:2]) + POINT_SOURCE_VIS_SIZE
    plt.xlim(-vis_axis_range, vis_axis_range)
    plt.ylim(-vis_axis_range, vis_axis_range)
    orbit_points = plt.scatter([], [], s=POINT_SOURCE_VIS_SIZE)
    vase_points = plt.scatter(*points[mask > 0][..., :2].T)
    plt.axis("off")
    plt.gca().set_aspect("equal")
    plt.gca().set_axis_off()
    plt.subplots_adjust(
        top=1,
        bottom=0,
        right=1,
        left=0,
        hspace=0,
        wspace=0,
    )
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Uncomment the line below for dark mode
    # plt.style.use("dark_background")

    # Generate the visualization
    with imageio.get_writer(
        utils.get_result_path("orbit_relighting.gif"),
        mode="I",
    ) as writer:
        num_steps = NUM_ORBITS * int((2 * np.pi) / np.linalg.norm(orbit.euler_step))
        for dt in range(0, num_steps, DELTA_T):
            print(f"Step: {dt+DELTA_T}/{num_steps}", end="\r")

            # Evaluate solution
            illuminated_img = programming.scene_relighting(
                mask,
                normals,
                albedo_img,
                points,
                light_source_xyz
                * IMG2LEC_COORDINATE_SYSTEM,  # reflect over x-z plane, reflect over x-y plane
                INTENSITY,
            )
            if illuminated_img.ndim == 2:
                illuminated_img = np.tile(illuminated_img[:, :, None], (1, 1, 3))
            illuminated_img = gamma_correction(
                illuminated_img, GAMMA, np.iinfo(np.uint8).max
            ).astype(np.uint8)

            # Create 2.5 dimensional visualization
            # Uncomment the line below to show shading on the vase in the orbit image. Very slow.
            # vase_points.set_color(illuminated_img[mask > 0] / np.iinfo(np.uint8).max)
            orbit_points.set_offsets(np.c_[light_source_xyz[0], light_source_xyz[1]])

            # Save plot to a memory buffer to load it as an array
            orbit_buf = io.BytesIO()
            plt.savefig(
                orbit_buf, format="png", bbox_inches="tight", pad_inches=1 / 4
            )  # save plot to buffer
            orbit_img = np.array(Image.open(orbit_buf))[
                ..., :3
            ]  # exclude alpha channel
            orbit_buf.close()

            orbit_img_scale = illuminated_img.shape[0] / orbit_img.shape[0]
            orbit_img_resized = cv2.resize(
                orbit_img, dsize=None, fx=orbit_img_scale, fy=orbit_img_scale
            )

            img = np.hstack((illuminated_img, orbit_img_resized))
            writer.append_data(img)
            cv2.imshow("Relighting", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            light_source_xyz = orbit.step(DELTA_T)

        print("\nSaving...")
    writer.close()


if __name__ == "__main__":
    REGISTRY = {
        "challenge1a": challenge1a,
        "challenge1b": challenge1b,
        "challenge1c": challenge1c,
        "challenge1d": challenge1d,
        "demo_surface_reconstruction": demo_surface_reconstruction,
        "challenge1e": challenge1e,
    }

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run_fn",
        type=str,
        nargs="+",
        choices=list(REGISTRY.keys()),
        help=f"The name of the functions that you would like to execute. Allowed values: {', '.join(REGISTRY)}",
        metavar="",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Shortcut to run all functions. Equivalent to '--run_fn {list of all functions}'.",
    )
    args = parser.parse_args()
    assert (
        args.run_fn is not None or args.run_all is True
    ), "Script expects arguments! See `python runHW.py -h` for help."

    if args.run_all:
        fns = list(REGISTRY.items())
    else:
        fns = [(fn_name, REGISTRY[fn_name]) for fn_name in args.run_fn]

    runHW(fns)
