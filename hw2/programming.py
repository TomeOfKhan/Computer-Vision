from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils


# --------------------------------------------------------------------------
# Walkthrough 1: Image processing
# --------------------------------------------------------------------------
def walkthrough1a():
    """
    Image processing: convolution, Gaussian smoothing

    Image credit: http://commons.wikimedia.org/wiki/File:Beautiful-pink-flower_-_West_Virginia_-_ForestWander.jpg
    """
    # Load the image
    img = utils.imread(utils.get_data_path("flower.png"))

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the sigma values
    sigma = [6, 12, 24]

    # Define the kernel sizes
    k = [int(np.ceil(2 * np.pi * s)) // 2 * 2 + 1 for s in sigma]

    # Create a figure with 2 rows and 2 columns of subplots
    fig, axs = plt.subplots(2, 2)

    # Display the original image
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Original")

    # Loop through the sigma values and apply Gaussian blur
    for i in range(len(sigma)):
        # Generate a Gaussian kernel
        h = cv2.GaussianBlur(img_gray, (k[i], k[i]), sigma[i])

        # Display the blurred image
        axs[(i + 1) // 2, (i + 1) % 2].imshow(h, cmap="gray")
        axs[(i + 1) // 2, (i + 1) % 2].axis("off")
        axs[(i + 1) // 2, (i + 1) % 2].set_title(f"sigma = {sigma[i]}")

    # Set the plot layout and save the figure
    fig.tight_layout()
    plt.savefig(utils.get_result_path("w1a_blur-flowers.png"))

    # Show result plot
    plt.show()


def walkthrough1b():
    """
    Edge detection

    Image credit: CAVE Lab
    """
    # Load the image
    img = utils.imread(utils.get_data_path("hello.png"))

    # Display the color image
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Color Image")

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    axs[0, 1].imshow(gray_img, cmap="gray")
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Grayscale Image")

    # Sobel edge detection
    thresh = 0.05 * np.max(gray_img)
    edge_img = cv2.Sobel(gray_img, cv2.CV_64F, 1, 1, ksize=3)
    edge_img[edge_img < thresh] = 0
    edge_img[edge_img >= thresh] = 255
    edge_img = edge_img.astype(np.uint8)
    axs[1, 0].imshow(edge_img, cmap="gray")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Sobel Edge Detection")

    # Canny edge detection
    thresh1 = 0.05 * np.max(gray_img)
    thresh2 = 0.1 * np.max(gray_img)
    edge_img = cv2.Canny(gray_img, thresh1, thresh2)
    axs[1, 1].imshow(edge_img, cmap="gray")
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Canny Edge Detection")

    # Save the resulting image
    plt.savefig(utils.get_result_path("w1b_hello-edges.png"))

    # Show result plot
    plt.show()


# --------------------------------------------------------------------------
# Challenge 1: Line Detection
# --------------------------------------------------------------------------
def find_edge_pixels(img: np.ndarray) -> np.ndarray:
    """
    Use the Canny edge detection function from OpenCV

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The edge image returned by Canny.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    return cv2.Canny(gray, 40, 100)


def generate_hough_accumulator(edge_img: np.ndarray) -> np.ndarray:
    """
    Generate the Hough accumulator for the edge image.

    Args:
        edge_img (np.ndarray): The edge image resultant from Canny edge detection.

    Returns:
        np.ndarray: The Hough accumulator (scaled from 0 to 255).
    """
    # Vote with rho = y*cos(theta) - x*sin(theta), using 1-pixel rho bins
    # and 1-degree theta bins in [0, 180). No NMS is applied here: the full
    # accumulator is returned for visualization. Duplicate line peaks are
    # suppressed in line_finder below (Challenge 1C).
    h, w = edge_img.shape
    diagonal = int(np.ceil(np.hypot(h, w)))
    angles = np.deg2rad(np.arange(180))
    y, x = np.nonzero(edge_img)
    accumulator = np.zeros((2 * diagonal + 1, 180), dtype=np.uint32)
    for j, theta in enumerate(angles):
        rho = np.rint(y * np.cos(theta) - x * np.sin(theta)).astype(int)
        accumulator[:, j] = np.bincount(rho + diagonal, minlength=2*diagonal+1)
    maximum = accumulator.max()
    return np.rint(accumulator * (255.0 / maximum)).astype(np.uint8) if maximum else accumulator.astype(np.uint8)


def line_finder(orig_img: np.ndarray, hough_img: np.ndarray) -> np.ndarray:
    """
    Find the lines in the original image using the Hough accumulator.

    Args:
        orig_img (np.ndarray): The original image to find lines in.
        hough_img (np.ndarray): The Hough accumulator.

    Returns:
        np.ndarray: The original image overlaid with lines.
    """
    output = orig_img.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    votes = hough_img.astype(float).copy()
    diagonal = (votes.shape[0] - 1) // 2
    y, x = np.nonzero(find_edge_pixels(orig_img))
    # Greedy peak selection: take the largest remaining accumulator bin,
    # stopping below max(60, 26% of the original peak), or after 35 peaks.
    # Draw only a sufficiently long, contiguous segment supported by edges.
    threshold = max(60, 0.26 * votes.max())


    for _ in range(35):
        row, col = np.unravel_index(np.argmax(votes), votes.shape)
        if votes[row, col] < threshold:
            break
        rho, theta = row - diagonal, np.pi * col / votes.shape[1]
        sn, cs = np.sin(theta), np.cos(theta)
        support = np.abs(x*sn-y*cs+rho) <= 2.5
        positions = np.sort(x[support]*cs+y[support]*sn)
        if len(positions) > 20:
            groups = np.split(positions, np.where(np.diff(positions)>15)[0]+1)
            group = max(groups, key=lambda g: g[-1]-g[0])
            if group[-1]-group[0] > 45:
                endpoints = [tuple(np.rint([-rho*sn+t*cs, rho*cs+t*sn]).astype(int)) for t in (group[0],group[-1])]
                cv2.line(output, endpoints[0], endpoints[1], (0, 0, 255), 2)
        # NMS zeros bins within +/-12 pixels in rho and +/-5 degrees in
        # theta around this peak, preventing adjacent bins from drawing the
        # same line repeatedly. At the 0/180-degree seam, wrap theta and
        # reverse rho because (rho, theta) and (-rho, theta+pi) describe
        # the same line. Only the working copy of the accumulator is changed.
        for dc in range(-5, 6):
            c = col + dc
            r = row if 0 <= c < votes.shape[1] else 2*diagonal-row
            votes[max(0,r-12):r+13, c % votes.shape[1]] = 0
    return output


# --------------------------------------------------------------------------
# Challenge 2: Coin Counting
# --------------------------------------------------------------------------
def generate_coin_hough_accumulator_2b(
    orig_img: np.ndarray,
    coins_info: Dict[int, Dict[str, Any]],
    radius_range: int = 0,
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """



    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) if orig_img.ndim == 3 else orig_img
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    small = gray[::2, ::2].copy()

    silhouette = (small > 60).astype(np.uint8) * 255
    edges = cv2.Canny(silhouette, 50, 100)
    gx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy) + 1e-8
    ys, xs = np.nonzero(edges)
    h, w = edges.shape
    if radius_range < 0 or int(radius_range) != radius_range:
        raise ValueError("radius_range must be a nonnegative integer")
    tolerance = int(radius_range)

    radii = sorted({r for nominal in coins_info
                    for r in range(nominal-tolerance, nominal+tolerance+1)
                    if r > 0})
    candidates = {}
    for radius in radii:
        r = radius / 2.0
        angles = np.linspace(0, 2*np.pi, int(np.ceil(2*np.pi*r)), endpoint=False)
        offsets = np.unique(np.rint(np.column_stack((r*np.cos(angles), r*np.sin(angles)))).astype(int), axis=0)
        plane = np.zeros(h*w, dtype=np.uint16)
        for dx, dy in offsets:
            cx, cy = xs-dx, ys-dy
            valid = (cx>=0)&(cx<w)&(cy>=0)&(cy<h)
            plane += np.bincount(cy[valid]*w+cx[valid], minlength=h*w).astype(np.uint16)
        plane = plane.reshape(h,w)

        pooled = sum(np.roll(np.roll(plane, dy, axis=0), dx, axis=1).astype(float) for dy in (-1,0,1) for dx in (-1,0,1))
        local = pooled.copy()
        for dy in range(-4,5):
            for dx in range(-4,5):
                local = np.maximum(local, np.roll(np.roll(pooled,dy,axis=0),dx,axis=1))
        yy, xx = np.nonzero((pooled==local)&(pooled > len(offsets)*0.65))


        phi = np.linspace(0, 2*np.pi, 180, endpoint=False)
        cosines, sines = np.cos(phi), np.sin(phi)
        for start in range(0, len(yy), 512):
            cy, cx = yy[start:start+512], xx[start:start+512]
            valid_center = (cx>=r+3)&(cy>=r+3)&(cx<w-r-3)&(cy<h-r-3)
            cy,cx = cy[valid_center],cx[valid_center]
            if not len(cx):
                continue
            supported = np.zeros((len(cx),len(phi)), dtype=bool)
            for dr in (-2,-1,0,1,2):
                px = np.rint(cx[:,None]+(r+dr)*cosines).astype(int)
                py = np.rint(cy[:,None]+(r+dr)*sines).astype(int)
                agreement = np.abs(gx[py,px]*cosines+gy[py,px]*sines)/magnitude[py,px]
                supported |= (edges[py,px]>0)&(agreement>0.80)
            vote_counts = supported.sum(axis=1)
            for x,y,votes in zip(cx,cy,vote_counts):
                if votes >= 76:
                    candidates[(int(2*x), int(2*y), int(radius))] = int(votes)
    return candidates


def coin_finder_2b(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    Find the coins using the Circle Hough accumulator.

    Args:
        coin_hough (Dict[Tuple[int, int, int], int]): The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.

    Returns:
        List[Tuple[int, int, int, int]]: Information of the detected coins -- a list of (x, y, r, votes).
    """




    result = []
    for (x,y,r),score in sorted(coin_hough.items(), key=lambda item:item[1], reverse=True):
        if score < 76:
            continue
        if all(np.hypot(x-a,y-b)>0.85*max(r,s) for a,b,s,_ in result):
            result.append((x,y,r,score))
    return result


def get_total_value_2b(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """
    total = 0
    for _,_,radius,_ in detected_coins:
        nominal = min(coins_info, key=lambda r:abs(r-radius))
        if abs(nominal-radius) <= radius_range:
            total += int(coins_info[nominal]['value'])
    return total / 100.0


def generate_coin_hough_accumulator_2c(
    orig_img: np.ndarray,
    coins_info: Dict[int, Dict[str, Any]],
    radius_range: int = 0,
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """

    return generate_coin_hough_accumulator_2b(orig_img, coins_info, radius_range)


def coin_finder_2c(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    Find the coins using the Circle Hough accumulator.

    Args:
        coin_hough (Dict[Tuple[int, int, int], int]): The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.

    Returns:
        List[Tuple[int, int, int, int]]: Information of the detected coins -- a list of (x, y, r, votes).
    """
    return coin_finder_2b(coin_hough)


def get_total_value_2c(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """
    return get_total_value_2b(coins_info, detected_coins, radius_range)


def generate_coin_hough_accumulator_2d(
    orig_img: np.ndarray,
    coins_info: Dict[int, Dict[str, Any]],
    radius_range: int = 0,
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """



    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) if orig_img.ndim == 3 else orig_img
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    small = gray[::2, ::2].copy()

    edges = cv2.Canny(small, 35, 85)
    gx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy) + 1e-8
    ys, xs = np.nonzero(edges)
    h, w = edges.shape
    if radius_range < 0 or int(radius_range) != radius_range:
        raise ValueError("radius_range must be a nonnegative integer")
    tolerance = int(radius_range)

    radii = sorted({r for nominal in coins_info
                    for r in range(nominal-tolerance, nominal+tolerance+1)
                    if r > 0})
    candidates = {}
    for radius in radii:
        r = radius / 2.0
        angles = np.linspace(0, 2*np.pi, int(np.ceil(2*np.pi*r)), endpoint=False)
        offsets = np.unique(np.rint(np.column_stack((r*np.cos(angles), r*np.sin(angles)))).astype(int), axis=0)
        plane = np.zeros(h*w, dtype=np.uint16)
        for dx, dy in offsets:
            cx, cy = xs-dx, ys-dy
            valid = (cx>=0)&(cx<w)&(cy>=0)&(cy<h)
            plane += np.bincount(cy[valid]*w+cx[valid], minlength=h*w).astype(np.uint16)
        plane = plane.reshape(h,w)

        pooled = sum(np.roll(np.roll(plane, dy, axis=0), dx, axis=1).astype(float) for dy in (-1,0,1) for dx in (-1,0,1))
        local = pooled.copy()
        for dy in range(-4,5):
            for dx in range(-4,5):
                local = np.maximum(local, np.roll(np.roll(pooled,dy,axis=0),dx,axis=1))
        yy, xx = np.nonzero((pooled==local)&(pooled > len(offsets)*0.65))


        phi = np.linspace(0, 2*np.pi, 180, endpoint=False)
        cosines, sines = np.cos(phi), np.sin(phi)
        for start in range(0, len(yy), 512):
            cy, cx = yy[start:start+512], xx[start:start+512]
            valid_center = (cx>=r+3)&(cy>=r+3)&(cx<w-r-3)&(cy<h-r-3)
            cy,cx = cy[valid_center],cx[valid_center]
            if not len(cx):
                continue
            supported = np.zeros((len(cx),len(phi)), dtype=bool)
            for dr in (-2,-1,0,1,2):
                px = np.rint(cx[:,None]+(r+dr)*cosines).astype(int)
                py = np.rint(cy[:,None]+(r+dr)*sines).astype(int)
                agreement = np.abs(gx[py,px]*cosines+gy[py,px]*sines)/magnitude[py,px]
                supported |= (edges[py,px]>0)&(agreement>0.80)
            vote_counts = supported.sum(axis=1)
            for x,y,votes in zip(cx,cy,vote_counts):
                if votes >= 120:
                    candidates[(int(2*x), int(2*y), int(radius))] = int(votes)
    return candidates


def coin_finder_2d(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    Find the coins using the Circle Hough accumulator.

    Args:
        coin_hough (Dict[Tuple[int, int, int], int]): The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.

    Returns:
        List[Tuple[int, int, int, int]]: Information of the detected coins -- a list of (x, y, r, votes).
    """

    strong_votes = {key: votes for key, votes in coin_hough.items() if votes >= 120}
    return coin_finder_2b(strong_votes)


def get_total_value_2d(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """
    return get_total_value_2b(coins_info, detected_coins, radius_range)
