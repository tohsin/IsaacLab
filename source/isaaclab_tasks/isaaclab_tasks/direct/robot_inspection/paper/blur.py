from pathlib import Path

import cv2
import numpy as np


PAPER_DIR = Path(__file__).resolve().parent
INPUT_PATH = PAPER_DIR / "UR10_RGB.png"
OUTPUT_PATH = PAPER_DIR / "inspection_blurred.png"

# A long, slightly angled kernel creates the streaking produced while a camera
# or object moves during exposure. Increase KERNEL_SIZE for a stronger effect.
KERNEL_SIZE = 41
MOTION_ANGLE_DEGREES = -8

image = cv2.imread(str(INPUT_PATH))
if image is None:
    raise FileNotFoundError(f"Could not read input image: {INPUT_PATH}")

kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
kernel[KERNEL_SIZE // 2, :] = 1.0
rotation = cv2.getRotationMatrix2D(
    (KERNEL_SIZE / 2 - 0.5, KERNEL_SIZE / 2 - 0.5),
    MOTION_ANGLE_DEGREES,
    1.0,
)
kernel = cv2.warpAffine(kernel, rotation, (KERNEL_SIZE, KERNEL_SIZE))
kernel /= kernel.sum()

blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

if not cv2.imwrite(str(OUTPUT_PATH), blurred):
    raise OSError(f"Could not write output image: {OUTPUT_PATH}")

print(f"Saved motion-blurred image to {OUTPUT_PATH}")
