# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:33:51 2024

@author : diwakar
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color

img = cv2.imread("images/grains2.jpg", 0)
cv2.imshow("Grains Image", img)
cv2.waitKey(0)

# Convert pixel to nanometer (nm)
pixels_to_nm = 0.5  # 1 pixel = 500 nm

ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

mask = dilated == 255

s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
labeled_mask, num_labels = ndimage.label(mask, structure=s)
img_label_overlay = color.label2rgb(labeled_mask, image=img, bg_label=0)
cv2.imshow("Labeled Grains", img_label_overlay)
cv2.waitKey(0)
clusters = measure.regionprops(labeled_mask, img)

print("\tArea(pixels)\tArea (nm^2)")

for prop in clusters:
    grain_area_pixels = prop.area
    grain_area_nm2 = grain_area_pixels * (pixels_to_nm ** 2)
    print('{}\t\t{}\t\t{}'.format(prop.label, grain_area_pixels, grain_area_nm2))

# Calculate grain size in nanometers (nm)
grain_sizes_nm = []

for prop in clusters:
    grain_diameter_pixels = np.sqrt(4*prop.area/np.pi) # Calculate radius in pixels
    grain_diameter_nm = grain_diameter_pixels * pixels_to_nm  # Convert radius to nm
    grain_size_nm = grain_diameter_nm

    grain_sizes_nm.append(grain_size_nm)



# Plot histogram with grain sizes in nm
plt.figure()
plt.hist(grain_sizes_nm, bins=50)
plt.xlabel('Grain Size(nm)')
plt.ylabel('Count')
plt.title('Grain Size Distribution')
plt.grid(True)
plt.show()

# Close all OpenCV windows
cv2.destroyAllWindows()