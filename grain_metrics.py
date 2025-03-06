#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Grain Metrics

This script uses Mask R-CNN to detect barley grains in images,
create masks, and calculate morphological parameters.
Results are saved in TSV format.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import time
import argparse
from skimage.measure import regionprops
from skimage.measure import find_contours
from math import sqrt, pi

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras.backend as K

# Import Mask-RCNN
try:
    import mrcnn.model as modellib
    from mrcnn.config import Config
    from mrcnn import utils
except ImportError:
    print("Error: Mask R-CNN library not found.")
    sys.exit(1)

# Configuration for Mask-RCNN
class InferenceConfig(Config):
    NAME = "seed"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + 1 seed
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 8192
    DETECTION_MIN_CONFIDENCE = 0.0  # Detect all instances
    RPN_NMS_THRESHOLD = 0.4
CONFIG = InferenceConfig()

CONFIG = InferenceConfig()
CLASS_NAMES = ['BG', 'SEED']

# Filtering functions using IQR
def filter_by_iqr_1(df):
    """Strict filtering for each sample."""
    parameters_to_filter = {
        'AS_seed_area': 1.0,
        'L_seed_length': 1.0,
        'W_seed_width': 1.0,
        'LWR_length_to_width_ratio': 1.2,
        'eccentricity': 1.2,
        'solidity': 1.2,
        'PL_perimeter_length': 1.0,
        'CS_seed_circularity': 1.5
    }
    
    for param, factor in parameters_to_filter.items():
        Q1 = df[param].quantile(0.25)
        Q3 = df[param].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[param] >= Q1 - factor * IQR) & (df[param] <= Q3 + factor * IQR)]
    return df

# Image Processing & Feature Extraction
def process_image(image_path, model):
    """Detect seeds and compute morphological parameters."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = model.detect([image], verbose=0)[0]
    
    data = []
    for obj in range(r['masks'].shape[-1]):
        mask = r['masks'][:, :, obj]
        score = r['scores'][obj]
        
        if score < 0.95:
            continue  # Skip low-confidence detections
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 100:
                continue  # Skip objects that are too small
            
            area = cv2.contourArea(contour)    #1. get seed area
            rect = cv2.minAreaRect(contour)    # Fit a minimum area rotated rectangle
            length = max(rect[1])    #2. get length
            width = min(rect[1])    #3. get width
            LWR = length / width    #4. get length/width ratio
            
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)    #5. get eccentricity
            
            hull = cv2.convexHull(contour)
            solidity = area / cv2.contourArea(hull)    #6. get solidity
            perimeter = cv2.arcLength(contour, True)    #7. get perimeter
            circularity = (4 * np.pi * area) / (perimeter ** 2)    #8. get circularity
            
            data.append([
                os.path.basename(image_path), obj, score, area, length, width,
                LWR, eccentricity, solidity, perimeter, circularity
            ])
    
    columns = ["file_name", "object_id", "detection_score", "AS_seed_area", "L_seed_length",
               "W_seed_width", "LWR_length_to_width_ratio", "eccentricity", "solidity",
               "PL_perimeter_length", "CS_seed_circularity"]
    return pd.DataFrame(data, columns=columns)

def process_images(input_dir, output_file, model):
    """Process all images, filter results, and save to a single table."""
    output_dir = os.path.dirname(output_file)
    
    df_list = []
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            df = process_image(os.path.join(input_dir, file), model)
            if not df.empty:
                df = filter_by_iqr_1(df).reset_index(drop=True)  # Apply first-level filtering
                df_list.append(df)
    
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(output_file, sep='\t', index=False)
    print(f"Results saved to {output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze barley seeds in images using Mask R-CNN')
    parser.add_argument('--input', required=True, help='Path to directory containing images to analyze')
    parser.add_argument('--output', required=True, help='Path to output file')
    parser.add_argument('--weights', default='data/barley/model_weights/mask_rcnn_barleyseeds_0040.h5', 
                        help='Path to Mask R-CNN weights file')
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        sys.exit(1)
    
    # Check if output file exists
    if os.path.exists(args.output):
        user_input = input(f"Warning: Output file {args.output} already exists. Overwrite? (y/n): ")
        if user_input.lower() != 'y':
            print("Process aborted by user.")
            sys.exit(1)
    
    # Check if weights file exists
    if not os.path.isfile(args.weights):
        print(f"Error: Weights file {args.weights} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Create model
    model = modellib.MaskRCNN(mode="inference", config=CONFIG, model_dir="")
    
    # Load weights
    print(f"Loading weights from {args.weights}")
    model.load_weights(args.weights, by_name=True)
    
    # Process images
    process_images(args.input, args.output, model)

if __name__ == "__main__":
    main()
