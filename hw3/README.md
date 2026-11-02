# Homework 3 - zk2335

All parts A-F are implemented and tested. The full runner completed successfully and generated eight result images. The three supplied camera photos are included unchanged in results/challenge_1f_input.

Run from this folder with NumPy, OpenCV, Matplotlib, and SciPy installed:

    python runHW.py --run_all

The supplied runner uses --run_all and --run_fn, which differ from the PDF's printed commands. For unattended runs, set MPLBACKEND=Agg. Cached manually selected points are in data/.cached_challenge1a.npy and data/.cached_challenge1b.npy; press Enter to accept them in interactive mode. Starter warnings about stty on Windows and noninteractive display do not affect the results.

Only programming.py was changed from the starter. All eight original functions, their signatures, and their docstrings are preserved. No helper functions were added. OpenCV calls are limited to remap for backward warping and resize for part F. Homography fitting uses np.linalg.eig, as requested. Feature matches use the provided utils.sift_matches.

For part F, images are named in spatial order, 01.jpg through 03.jpg. The program begins with the middle view, 02.jpg, then adds 01.jpg and 03.jpg, reducing perspective stretching. Inputs are resized to fit 1280x720 while preserving aspect ratio. The panorama is 1621x1144 pixels. Black borders are uncovered canvas; minor traffic ghosting reflects objects moving between photographs.

The archive hw3_zk2335.zip contains programming.py and results/, including all three original camera photographs and the completed panorama. It has not been uploaded to CourseWorks.
