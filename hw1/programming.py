import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils


# --------------------------------------------------------------------------
# Walkthrough 1: Basic Numpy
# --------------------------------------------------------------------------
def walkthrough1():
    """
    Basic operations in NumPy.
    """
    x = np.arange(6)
    print("x =", x)
    print()

    # Indexing
    print("Indexing")
    print("x[1] =", x[1])
    print("x[1:4] =", x[1:4])
    print("x[::-1] =", x[::-1])  # Index backwards
    print("x[::2] =", x[::2])  # Index every other element, starting at 0
    print("x[1::2] =", x[1::2])  # Index every other element, starting at 1
    print()

    # Arithmetic operations
    print("Add a constant to every element:")
    print("x + 10 =", x + 10)
    print()

    print("Element-wise addition")
    y = np.asarray([0, 1, 0, 1, 0, 1])
    print("y =", y)
    print("x + y =", x + y)
    print(x + y)
    print()

    # Create a 3x3 array and specify the datatype is 32-bit floating point.
    A = np.asarray([[1, 2, 3], [-2, 3, 4], [4, 3, 2]], dtype=np.float32)

    print("A:")
    print(A)
    print()

    # Unravel the elements into a vector (row-major order)
    print("A.ravel():")
    print(A.ravel())
    print()

    # Compute the transpose
    print("Transpose of A:")
    print(A.T)
    print()

    # Compute the matrix inverse
    A_inv = np.linalg.inv(A)
    print("Inverse of A: ")
    print(A_inv)
    print()

    # In NumPy, the operators "*" and "@" function differently for matrices.
    # "@" performs a matrix multiplication, and "*" performs an element-wise
    # operation

    # Compute the matrix multiplication A^(-1) A using the "@" operator
    print("A_inv @ A:")
    print(A_inv @ A)
    print()

    # Compute the element-wise operation using the "*" operator
    print("A_inv * A:")
    print(A_inv * A)
    print()

    # Compute the eigenvalues and eigenvectors
    h, v = np.linalg.eig(A)
    print("Eigenvalues of A:")
    print(h)
    print("Eigenvectors of A")
    print(v)
    print()


# --------------------------------------------------------------------------
# Walkthrough 2: Basic image operations
# --------------------------------------------------------------------------
def walkthrough2():
    """
    Basic image operations.
    """
    # Load an image
    img = utils.imread(utils.get_data_path("cameraman.png"))
    utils.imshow(img, "Camera man")  # Hover over pixels to see each pixel value

    # Check its datatype
    print("Image datatype", img.dtype)
    print("Image shape", img.shape)
    print("Image height", img.shape[0])
    print("Image width", img.shape[1])

    # Cast image to double
    img = img.astype(np.float32) / 255
    # Hover over pixels to see their values are now in [0, 1]
    utils.imshow(img, title="Camera man, double")

    ## Load a color image
    img = utils.imread(utils.get_data_path("van-gogh.png"))
    utils.imshow(img, title="Vincent Van Gogh")
    print("Image shape", img.shape)

    # Convert image to black-and-white
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    utils.imshow(img_bw, title="Vincent Van Gogh, Black and White")
    print("Black-and-white image shape", img_bw.shape)

    ## Create two matrices holding the (row, column) coordinates of an image.
    img_size = img.shape
    r, c = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]), indexing="ij")

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(r, cmap="gray")
    axs[0].set_title("Row Coordinate")
    axs[1].imshow(c, cmap="gray")
    axs[1].set_title("Column Coordinate")
    plt.show()


# --------------------------------------------------------------------------
# Walkthrough 3: Van Gogh collage
# --------------------------------------------------------------------------
def walkthrough3():
    """
    Create a Vincent Van Gogh collage.
    Fill in code marked with ...
    """
    # Load the image "van-gogh.png"
    img = utils.imread(utils.get_data_path("van-gogh.png"))  # TODO

    # Note the image is a uint8 type and the maximum pixel value is 255
    print("img.dtype:", img.dtype)
    print("img.max:", img.max())

    # uint8 is memory efficient. Since we will perform some arithmetic
    # operations on the image, uint8 needs to be used with caution. Let's
    # case the image to double.
    img = img.astype(np.float64) / 255
    print("img.dtype:", img.dtype)
    print("img.max():", img.max())

    # Display the image
    utils.imshow(img, title="Original Image")

    # Separate the image into three color channels and store each channel in
    # a new image
    # By default, the functions in utils.py read and display images in BGR order.
    # This means the blue channel is index 0, the green channel is index 1, and
    # the red channel is index 2 along the third dimension.
    red_channel = img[:, :, 2]
    utils.imshow(red_channel, title="Red Channel")
    red_img = np.zeros(img.shape)
    red_img[:, :, 2] = red_channel
    utils.imshow(red_img, title="Red Image")

    # Similarly extract green_channel and blue_channel and create green_image
    # and blue_image
    green_channel = img[:, :, 1]
    utils.imshow(red_channel, title="Green Channel")
    green_img = np.zeros(img.shape)
    green_img[:, :, 1] = green_channel
    utils.imshow(green_img, title="Green Image")

    blue_channel = img[:, :, 0]
    utils.imshow(blue_channel, title="Blue Channel")
    blue_img = np.zeros(img.shape)
    blue_img[:, :, 0] = blue_channel
    utils.imshow(blue_img, title="Blue Image")
     # TODO

    # Create a 1x4 image collage in the following format:
    # original image | red channel | green channel | blue channel
    collage_1x4 = np.concatenate((img, red_img, green_img, blue_img), axis=1)
    utils.imshow(collage_1x4, title="1x4 Collage")

    # Now create a 2 x 2 image collage in the following arrangement
    #
    # original image | red channel
    # green channel  | blue channel

    top_row = np.concatenate((img,red_img), axis=1)
    bottom_row = np.concatenate((green_img,blue_img), axis=1)
    collage_2x2 = np.concatenate((top_row,bottom_row), axis=0)  # TODO
    utils.imshow(collage_2x2, title="2x2 Collage")

    # Save the collage as "collage.png"
    utils.imwrite(utils.get_result_path("collage.png"), collage_2x2)  # TODO


# --------------------------------------------------------------------------
# Walkthrough 4: I <3 NY image overlay
# --------------------------------------------------------------------------
def walkthrough4():
    """
    Create an I <3 NY image.
    Fill in code marked with ...
    """
    # Load the image "I_Love_New_York.png"
    img_ilnyc = utils.imread(utils.get_data_path("I_Love_New_York.png"))  # TODO
    utils.imshow(img_ilnyc, "Original Image")

    # Convert to grayscale
    img_ilnyc_gray = cv2.cvtColor(img_ilnyc, cv2.COLOR_BGR2GRAY)
    utils.imshow(img_ilnyc_gray, "Gray Scale")

    # Convert the image into a binary mask using a threshold value
    threshold = 0.5  # TODO
    binary_mask = (img_ilnyc_gray > threshold).astype(np.uint8) * 255

    # Load the image "nyc.png"
    img_nyc = utils.imread(utils.get_data_path("nyc.png"))  # TODO

    # Resize the nyc.png image so that its height is 500 pixels
    height = img_nyc.shape[0]  # TODO
    width = img_nyc.shape[1]  # TODO
    scale = 500 / height
    small_nyc = cv2.resize(img_nyc, (int(width * scale), int(height * scale)))

    # Resize the ILoveNY mask so that its height is 400 pixels
    scale = 400 / img_ilnyc.shape[0]
    resized_mask = cv2.resize(
        binary_mask, (int(img_ilnyc.shape[1] * scale), int(img_ilnyc.shape[0] * scale))
    )
    utils.imshow(resized_mask, title="Resized Mask")

    # Cast the mask to a bool
    resized_mask = resized_mask.astype(np.bool_)

    # Note that small_nyc and iresized_mask have different heights and widths
    print("small_nyc.shape", small_nyc.shape)
    print("iresized_mask.shape", resized_mask.shape)

    # No worries. Let's use the collage technique learned in walkthrough3 to
    # make resized_mask have the same height and width as small_nyc
    height_diff = small_nyc.shape[0] - resized_mask.shape[0]
    width_diff = small_nyc.shape[1] - resized_mask.shape[1]
    mask_height = resized_mask.shape[0]
    utils.imshow(resized_mask.astype(np.uint8) * 255)

    # Pad the left and right sides of iresized_height
    padded_mask = np.concatenate(
        (
            np.zeros((mask_height, width_diff // 2), dtype=np.bool_),
            resized_mask,
            np.zeros((mask_height, width_diff // 2), dtype=np.bool_),
        ),
        axis=1,
    )
    utils.imshow(padded_mask.astype(np.uint8) * 255, title="Pad left and right")

    # Pad the top and bottom sides
    padded_mask = np.concatenate(
        (
            np.zeros((height_diff//2, padded_mask.shape[1]), dtype=np.bool_),
            padded_mask,
            np.zeros((height_diff//2, padded_mask.shape[1]), dtype=np.bool_),
        ),
        axis=0,
    )  # TODO
    utils.imshow(padded_mask.astype(np.uint8) * 255, title="Pad top and bottom")

    """
    # NumPy has many convenient functions. The above code can actually be done
    # with a single line.
    # The NumPy documentation is a good place to discover what tools are 
    # available to you.

    ipadded_mask = np.pad(resized_mask, 
                          ((height_diff//2, height_diff//2), 
                           (width_diff//2, width_diff//2)))
    """

    # Now let's burn the I <3 NY logo into the Manhattan scene
    love_small_nyc = small_nyc.copy()
    # Set the red channel
    # The color channels are in BGR order, so red is the last channel


    red_channel = love_small_nyc[:, :, 2]
    red_channel[padded_mask] = 255

    blue_channel = love_small_nyc[:, :, 0]
    blue_channel[padded_mask] = 0  # TODO: Set the blue channel

    green_channel = love_small_nyc[:, :, 1]
    green_channel[padded_mask] = 0  # TODO: Set the green channel

    utils.imshow(love_small_nyc)

    # Save the collage as output_nyc.png
    utils.imwrite(utils.get_result_path("output_nyc.png"), love_small_nyc)  # TODO

    
