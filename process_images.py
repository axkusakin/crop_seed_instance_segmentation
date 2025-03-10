#!/usr/bin/env python3
"""
Image Preprocessing for Grain Metrics

This script preprocesses barley seed images before Mask R-CNN detection.
It enhances contrast, reduces noise, and converts images to grayscale 
to improve seed segmentation accuracy.

Processed images are saved in the specified output folder.
"""

import os
import cv2
import argparse
import numpy as np

def preprocess_image(input_path, output_path):
    """Apply preprocessing steps to an image and save the result."""
    image = cv2.imread(input_path)  # Load the original color image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)  # Keep original grayscale values

    # Save processed image
    cv2.imwrite(output_path, enhanced)
    print(f"Saved: {output_path}")

def process_folder(input_dir, output_dir):
    """Process all .png images in the input folder and save them in the output folder."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            preprocess_image(input_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Preprocess images before seed recognition")
    parser.add_argument("-i", "--input", required=True, help="Path to input folder containing .png images")
    parser.add_argument("-o", "--output", required=True, help="Path to output folder for preprocessed images")
    args = parser.parse_args()

    process_folder(args.input, args.output)

if __name__ == "__main__":
    main()
