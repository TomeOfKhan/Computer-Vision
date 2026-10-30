import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils

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
# Challenge 1: Image Mosaicking App
# --------------------------------------------------------------------------
def challenge1a():
    # Load images
    src_img = utils.imread(
        utils.get_data_path("portrait.png"), normalize=True, rgb=True
    )
    dest_img = utils.imread(
        utils.get_data_path("portrait_transformed.png"), normalize=True, rgb=True
    )

    # Load cached points
    cache_path = Path(utils.get_data_path(".cached_challenge1a.npy")).resolve()
    if cache_path.exists():
        src_cached_points, dest_cached_points, test_cached_points = np.load(cache_path)
    else:
        src_cached_points, dest_cached_points, test_cached_points = [], [], []

    # Point correspondences used to compute the homography
    src_pts = utils.click_pts(
        src_img, 4, "Click first 4 points...", prev_points=src_cached_points
    )
    dest_pts = utils.click_pts(
        dest_img, 4, "Click corresponding 4 points...", prev_points=dest_cached_points
    )
    H = programming.compute_homography(src_pts, dest_pts)

    # Choose a new set for testing the homography
    test_pts = utils.click_pts(
        src_img, 4, "Click 4 different test points..", prev_points=test_cached_points
    )
    test_pts_dest = programming.apply_homography(H, test_pts)

    # Cache selected points
    np.save(cache_path, np.stack((src_pts, dest_pts, test_pts), axis=0))

    # Display and save the test point correspondences
    fig = utils.show_correspondences(
        src_img, dest_img, test_pts, test_pts_dest, "Test Point Correspondences"
    )
    fig.tight_layout()
    fig.savefig(utils.get_result_path("1a.png"), dpi=192)
    plt.show()


def challenge1b():
    bg_img = utils.imread(utils.get_data_path("Osaka.png"), rgb=True, normalize=True)
    portrait_img = utils.imread(
        utils.get_data_path("portrait_small.png"), rgb=True, normalize=True
    )

    # Load cached points
    cache_path = Path(utils.get_data_path(".cached_challenge1b.npy")).resolve()
    if cache_path.exists():
        portrait_cached_points, bg_cached_points = np.load(cache_path)
    else:
        portrait_cached_points, bg_cached_points = [], []

    # Click on point correspondences
    border = 25
    portrait_img_padded = cv2.copyMakeBorder(
        portrait_img, *[border] * 4, cv2.BORDER_CONSTANT, value=[1.0] * 3
    )
    portrait_pts = utils.click_pts(
        portrait_img_padded, 4, prev_points=portrait_cached_points
    )
    bg_pts = utils.click_pts(bg_img, 4, prev_points=bg_cached_points)

    # Cache selected points
    np.save(cache_path, np.stack((portrait_pts, bg_pts), axis=0))

    result = programming.warp_img_onto(
        portrait_img, bg_img, portrait_pts - border, bg_pts
    )
    result = np.clip(result, 0, 1)  # clip to valid float image range

    # Display and save the result
    utils.imshow(result, flag=None)
    utils.imwrite(utils.get_result_path("1b.png"), result, cv2.COLOR_RGB2BGR)


def challenge1c():
    img_s = utils.imread(utils.get_data_path("law-left.jpg"), rgb=True, normalize=True)
    img_d = utils.imread(
        utils.get_data_path("law-center.jpg"), rgb=True, normalize=True
    )

    src_pts, dest_pts = utils.sift_matches(img_s, img_d)

    # Display and save the correspondences
    fig = utils.show_correspondences(
        img_s, img_d, src_pts, dest_pts, title="Before RANSAC", show_every_n=4
    )
    fig.tight_layout()
    fig.savefig(utils.get_result_path("1c_before-ransac.png"), dpi=192)
    plt.show()

    # Run RANSAC using parameters defined at the top of programming.py
    inlier_idx, H = programming.run_RANSAC(
        src_pts,
        dest_pts,
        programming.CHALLENGE_1C_RANSAC_N,
        programming.CHALLENGE_1C_RANSAC_EPS,
    )

    # Display and save the correspondences
    fig = utils.show_correspondences(
        img_s,
        img_d,
        src_pts[inlier_idx, :],
        dest_pts[inlier_idx, :],
        title="After RANSAC",
        show_every_n=4,
    )
    fig.tight_layout()
    fig.savefig(utils.get_result_path("1c_after-ransac.png"), dpi=192)
    plt.show()


def challenge1d():
    fish, fish_mask = utils.imread_alpha(
        utils.get_data_path("escher_fish.png"), normalize=True
    )
    horse, horse_mask = utils.imread_alpha(
        utils.get_data_path("escher_horsemen.png"), normalize=True
    )

    img_overlay = programming.blend_image_pair(
        fish, fish_mask, horse, horse_mask, "overlay"
    )
    img_blend = programming.blend_image_pair(fish, fish_mask, horse, horse_mask, "blend")

    utils.imshow(img_overlay, "Overlay", flag=None)
    utils.imshow(img_blend, "Blend", flag=None)

    utils.imwrite(utils.get_result_path("1d_overlay.png"), img_overlay)
    utils.imwrite(utils.get_result_path("1d_blend.png"), img_blend)


def challenge1e():
    # Read images
    imgs = []
    files = [
        utils.get_data_path("law-center.jpg"),
        utils.get_data_path("law-left.jpg"),
        utils.get_data_path("law-right.jpg"),
    ]
    for fname in files:
        img = utils.imread(fname, normalize=True)
        imgs.append(img)

    # Stitch images
    panorama = programming.stitch_imgs(imgs)

    panorama = np.clip(panorama, 0, 1)
    utils.imshow(panorama)
    utils.imwrite(utils.get_result_path("1e.png"), panorama)


def challenge1f():
    # Panorama on your own images.
    panorama = programming.build_your_own_panorama()

    panorama = np.clip(panorama, 0, 1)
    utils.imshow(panorama, flag=None)
    utils.imwrite(utils.get_result_path("1f.png"), panorama, flag=cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    REGISTRY = {
        "challenge1a": challenge1a,
        "challenge1b": challenge1b,
        "challenge1c": challenge1c,
        "challenge1d": challenge1d,
        "challenge1e": challenge1e,
        "challenge1f": challenge1f,
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
    ), "Script expects arguments! See `python runHW{...}.py -h` for help."

    if args.run_all:
        fns = list(REGISTRY.items())
    else:
        fns = [(fn_name, REGISTRY[fn_name]) for fn_name in args.run_fn]

    runHW(fns)
