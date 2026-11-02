# Homework 2

The provided PDF contains only a programming assignment.

## Run

From this folder, with NumPy, Matplotlib, OpenCV, and SciPy installed:

    python runHW.py --run_all

The supplied runner accepts `--run_all` and `--run_fn`; its arguments differ from the commands printed in the assignment PDF. For unattended execution, set MPLBACKEND=Agg. Plot-display warnings in this mode are expected. On Windows the supplied banner helper also prints an innocuous missing `stty` warning.

Only `programming.py` was changed from the programming starter. `runHW.py`, `utils.py`, and the data files are unchanged. The implementation uses only permitted OpenCV calls and NumPy for both Hough transforms.

## Results

- Challenge 2B: 16 coins, $2.05 (reference: 16 coins, $2.05).
- Challenge 2C: 24 coins, $2.90 (reference: 24 coins, $2.90).
- Challenge 2D: 24 coins, $2.90 (reference: 24 coins, $2.90).

The saved overlays were visually checked. These results concern the supplied images; performance on other images has not been evaluated. The radius_range argument is honored exactly: zero searches only the calibrated radii. Circle accumulator values are integer support-vote counts from 180 angular boundary bins. Greedy non-maximum suppression removes duplicate circles.

## Submission

`hw2_zk2335.zip` contains exactly `programming.py` and `results/`, as requested in the PDF. The starter files are available separately in this folder. The archive has not been uploaded to CourseWorks.

