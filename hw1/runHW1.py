# If any of packages fail to import, it is like that you have not correctly
# installed the course anaconda environment.
import argparse
from typing import Callable, List, Tuple

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
    python runHW -h                                : show help and list all the registered functions
    python runHW --run_fn walkthrough1             : execute a specific function
    python runHW --run_fn walkthrough1 challenge1a : execute a set of specific functions
    python runHW --run_all                         : execute all the registered functions
    """
    np.set_printoptions(precision=4, suppress=True)

    for fn_name, fn_callable in fns:
        utils.print_banner(fn_name)
        fn_callable()
        utils.print_banner()
        print()


if __name__ == "__main__":
    REGISTRY = {
        "walkthrough1": programming.walkthrough1,
        "walkthrough2": programming.walkthrough2,
        "walkthrough3": programming.walkthrough3,
        "walkthrough4": programming.walkthrough4,
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
