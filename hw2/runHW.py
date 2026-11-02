# If any of packages fail to import, it is like that you have not correctly
# installed the course anaconda environment.
import argparse
from typing import Callable, List, Tuple

import cv2
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
# Walkthrough 1: Image processing
# --------------------------------------------------------------------------
def walkthrough1():
    programming.walkthrough1a()
    programming.walkthrough1b()


# --------------------------------------------------------------------------
# Challenge 1: Line Detection
# --------------------------------------------------------------------------
def challenge1a():
    img_list = ["line_1", "line_2", "line_3", "line_4", "line_5"]

    for i in range(len(img_list)):
        img = utils.imread(
            utils.get_data_path(f"{img_list[i]}.png"), cv2.IMREAD_GRAYSCALE
        )
        edge_img = programming.find_edge_pixels(img)

        # Save the edge image
        utils.imwrite(utils.get_result_path(f"1a_edge-{img_list[i]}.png"), edge_img)

        # Display the image
        utils.imshow(edge_img)


def challenge1b():
    img_list = ["line_1", "line_2", "line_3", "line_4", "line_5"]

    for i in range(len(img_list)):
        img = utils.imread(
            utils.get_result_path(f"1a_edge-{img_list[i]}.png"), cv2.IMREAD_GRAYSCALE
        )
        hough_accumulator = programming.generate_hough_accumulator(img)

        # We'd like to save the hough accumulator array as an image to
        # visualize it. The values should be between 0 and 255 and the
        # data type should be uint8.
        utils.imwrite(
            utils.get_result_path(f"1b_accumulator-{img_list[i]}.png"),
            hough_accumulator,
        )

        # Display the image
        utils.imshow(hough_accumulator)


def challenge1c():
    img_list = ["line_1", "line_2", "line_3", "line_4", "line_5"]

    for i in range(len(img_list)):
        orig_img = utils.imread(utils.get_data_path(f"{img_list[i]}.png"))
        hough_img = utils.imread(
            utils.get_result_path(f"1b_accumulator-{img_list[i]}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        line_img = programming.line_finder(orig_img, hough_img)

        # Save the line image to file
        utils.imwrite(utils.get_result_path(f"1c_line-{img_list[i]}.png"), line_img)

        # Display the image
        utils.imshow(line_img)


# --------------------------------------------------------------------------
# Challenge 2: Coin Counting
# --------------------------------------------------------------------------
def challenge2a():
    # Get the radii of coins and print to stdout
    utils.get_radii(debug=True)


def challenge2b():
    # Get the radii of coins
    coins_info = utils.get_radii(debug=False)

    # Circle Hough Transform
    img_name = "coins_2"
    img = utils.imread(utils.get_data_path(f"{img_name}.png"))
    coin_hough = programming.generate_coin_hough_accumulator_2b(img, coins_info)
    detected_coins = programming.coin_finder_2b(coin_hough)
    coin_img = utils.annotate_coins(img, detected_coins, coins_info, radius_range=10)
    total_value = programming.get_total_value_2b(coins_info, detected_coins)
    print(f"Total value detected: ${total_value}")
    print(f"Actual total value: $2.05")

    # Save the circle image
    utils.imwrite(utils.get_result_path(f"2b_coins-{img_name}.png"), coin_img)

    # Display the image
    utils.imshow(coin_img)


def challenge2c():
    # Get the radii of coins
    coins_info = utils.get_radii(debug=False)

    # Circle Hough Transform
    img_name = "coins_3"
    img = utils.imread(utils.get_data_path(f"{img_name}.png"))
    coin_hough = programming.generate_coin_hough_accumulator_2c(img, coins_info)
    detected_coins = programming.coin_finder_2c(coin_hough)
    coin_img = utils.annotate_coins(img, detected_coins, coins_info, radius_range=10)
    total_value = programming.get_total_value_2c(coins_info, detected_coins)
    print(f"Total value detected: ${total_value}")
    print(f"Actual total value: $2.9")

    # Save the circle image
    utils.imwrite(utils.get_result_path(f"2c_coins-{img_name}.png"), coin_img)

    # Display the image
    utils.imshow(coin_img)


def challenge2d():
    # Get the radii of coins
    coins_info = utils.get_radii(debug=False)

    # Circle Hough Transform
    img_name = "coins_4"
    img = utils.imread(utils.get_data_path(f"{img_name}.png"))
    coin_hough = programming.generate_coin_hough_accumulator_2d(img, coins_info)
    detected_coins = programming.coin_finder_2d(coin_hough)
    coin_img = utils.annotate_coins(img, detected_coins, coins_info, radius_range=10)
    total_value = programming.get_total_value_2d(coins_info, detected_coins)
    print(f"Total value detected: ${total_value}")
    print(f"Actual total value: $2.9")

    # Save the circle image
    utils.imwrite(utils.get_result_path(f"2d_coins-{img_name}.png"), coin_img)

    # Display the image
    utils.imshow(coin_img)


if __name__ == "__main__":
    REGISTRY = {
        "walkthrough1": walkthrough1,
        "challenge1a": challenge1a,
        "challenge1b": challenge1b,
        "challenge1c": challenge1c,
        "challenge2a": challenge2a,
        "challenge2b": challenge2b,
        "challenge2c": challenge2c,
        "challenge2d": challenge2d,
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
