import cv2
import numpy as np

# Load an image
img = cv2.imread('image_data/train/correct_assembly_yellow_01_std1.jpg')

# Get dimensions
height, width = img.shape[:2]
print(f"Original size: {width}×{height}")

# Adjust these values until crop looks good
crop_x = int(width * 0.15)   # Start at 15% from left
crop_y = int(height * 0.02)  # Start at 5% from top
crop_w = int(width * 0.6)    # Width = 70% of total
crop_h = int(height * 0.6)   # Height = 90% of total

# Crop
cropped = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
resized = cv2.resize(cropped, (384, 384))

# Show both
cv2.imshow('Original', img)
cv2.imshow('Cropped', cropped)
cv2.imshow('Resized', resized)

print(f"Cropped size: {cropped.shape[1]}×{cropped.shape[0]}")
print("Adjust crop_x, crop_y, crop_w, crop_h values and re-run")

cv2.waitKey(0)
cv2.destroyAllWindows()
