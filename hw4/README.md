# Homework 4 - zk2335

## Submission

hw4_zk2335.zip contains exactly written.pdf, programming.py, and results/. It has not been uploaded to CourseWorks or GitHub.

The written solution proves the effective-light formula for equal and unequal source intensities. Editable LaTeX is in written-source/main.tex. The PDF compiled successfully in the built-in editor and was rendered and visually checked.

## Programming

Only programming.py was edited from the starter. It retains all five original functions, argument lists, return annotations, and docstrings. No extra functions or imports were added. Its implementation uses NumPy only. The original runHW.py and utils.py are unchanged.

The comments in compute_light_directions justify the sphere normal, the right-handed image coordinate convention, the diffuse brightness peak, and intensity scaling, as required by the PDF. Background normals point toward the camera (0, 0, -1). The foreground mask requires brightness above 5 gray levels in at least three images. Photometric stereo uses all adequately illuminated sources, with at least the three brightest, and solves their overdetermined system while retaining unequal source intensities. Albedo is scaled by one global constant.

## Run

From this directory with NumPy, OpenCV, Matplotlib, Pillow, and imageio installed:

    python runHW.py --run_all

The supplied runner accepts --run_all and --run_fn, which differ from the PDF's printed commands. Close the Matplotlib figures when finished viewing them, including the 3D reconstruction demo, so execution can proceed to the relighting stage. The original banner helper prints harmless missing-stty messages on Windows.

For automated testing, all runner stages were executed with noninteractive display callbacks: Matplotlib show closed displayed figures and OpenCV imshow/waitKey were disabled. These were test-harness changes in memory only, not edits to either starter file or submitted code. Closing the demo figure avoids a starter plotting issue where the next 2D scatter plot otherwise reuses the 3D axes under a noninteractive backend.

## Verification

All seven runner result files are present, including the 90-frame orbit_relighting.gif. Result images and multiple animation frames were visually checked. Synthetic numerical checks cover the sphere centroid/radius, foreground mask threshold, unequal lighting intensities, cast-shadow rejection, unit normals, albedo normalization, inverse-square radiance, the 1/pi factor, negative-dot clamping, and uint8 output.

As a separate calibration check, recovered normals on the interior of the calibration sphere have a median angular error of 1.08 degrees and a 95th percentile error of 1.52 degrees against its known geometry. See validation.json for details.
