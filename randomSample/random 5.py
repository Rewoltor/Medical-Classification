#!/usr/bin/env python3
"""
Random 10 Sampling Script
========================

Extracts 10 random images from class 1 (KL Grade 1).
Maps to filenames 51-60.
Generates a CSV log with ALL columns from the original predictions.csv.
"""

import os
import random
import shutil
import csv
import sys
from typing import Dict, List

# Configuration
random.seed(42)  # Reproducibility

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PREDICTIONS_CSV = os.path.join(ROOT_DIR, 'predicted', 'predictions.csv')
SOURCE_ORIGINAL_BASE = os.path.join(ROOT_DIR, 'dataset', 'test')

DEST_DIR = os.path.join(BASE_DIR, 'sampled_5')
NEW_PREDICTIONS_CSV = os.path.join(DEST_DIR, 'predictions.csv')

# Output filename mapping: Class -> List of New Filenames (without extension)
# 10 images total from KL Grade 1
# Class 1 -> 51 to 60
CLASS_TO_FILENAMES = {
    '1': ['51', '52', '53', '54', '55', '56', '57', '58', '59', '60']
}

def load_all_predictions() -> Dict[str, List[Dict[str, str]]]:
    """Load all predictions grouped by class (ground_truth_raw)."""
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"❌ Error: Predictions CSV not found: {PREDICTIONS_CSV}")
        sys.exit(1)

    grouped = {k: [] for k in CLASS_TO_FILENAMES.keys()}
    
    with open(PREDICTIONS_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kl_grade = row['ground_truth_raw'] # This is the class
            
            if kl_grade in grouped:
                grouped[kl_grade].append(row)
                
    return grouped

def main():
    print("="*60)
    print("RANDOM 10 SAMPLING SCRIPT (Class 1)")
    print("="*60)

    # 1. Setup Destination
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)
    print(f"Created directory: {DEST_DIR}")

    # 2. Load Data
    print("Loading predictions.csv...")
    grouped_data = load_all_predictions()

    selected_rows = []
    
    # 3. Sample and Copy
    print("\nSampling images...")
    for class_id in sorted(CLASS_TO_FILENAMES.keys()):
        rows = grouped_data[class_id]
        target_filenames = CLASS_TO_FILENAMES[class_id]
        count_needed = len(target_filenames)
        
        if len(rows) < count_needed:
            print(f"⚠️  Warning: Not enough images for Class {class_id} in CSV. Needed {count_needed}, found {len(rows)}.")
            continue
            
        # Randomly select N unique images
        selected_class_rows = random.sample(rows, count_needed)
        
        for i, selected_row in enumerate(selected_class_rows):
            new_name_base = target_filenames[i]
            
            # Original details
            orig_image_path_in_csv = selected_row['image']
            filename = os.path.basename(orig_image_path_in_csv)
            
            # Construct full source path
            source_path = os.path.join(SOURCE_ORIGINAL_BASE, class_id, filename)
            
            if not os.path.exists(source_path):
                source_path_alt = os.path.join(ROOT_DIR, orig_image_path_in_csv)
                if os.path.exists(source_path_alt):
                    source_path = source_path_alt
                else:
                    print(f"❌ Error: Image file not found: {source_path}")
                    continue

            # Determine new filename
            ext = os.path.splitext(filename)[1]
            new_filename = f"{new_name_base}{ext}"
            
            dest_path = os.path.join(DEST_DIR, new_filename)
            
            # Copy
            shutil.copy2(source_path, dest_path)
            print(f"  Class {class_id}: {filename} -> {new_filename}")
            
            # Prepare row for new CSV
            new_row = selected_row.copy()
            
            # Calculate ai_confidence
            try:
                prob = float(selected_row.get('probability', 0.5))
                ai_conf = abs(prob - 0.5) * 2
            except ValueError:
                ai_conf = 0.0
                
            # Update fields
            new_row['image'] = new_filename
            
            if 'overlay' in selected_row and selected_row['overlay'].strip():
                gradcam_name = os.path.basename(selected_row['overlay'])
            else:
                name_no_ext = os.path.splitext(filename)[0]
                gradcam_name = f"{name_no_ext}_gradcam.png"
                
            new_row['image_name'] = gradcam_name
            new_row['image_name_original'] = filename
            new_row['class'] = class_id 
            new_row['ai_confidence'] = f"{ai_conf:.4f}"
            
            selected_rows.append(new_row)

    # 4. Generate CSV
    print("\nGenerating CSV log...")
    
    # Columns from randomSample.py
    fieldnames = [
        'image',
        'image_name',
        'image_name_original',
        'class',
        'ai_confidence',
        'ground_truth_raw',
        'ground_truth_binary',
        'output_logit',
        'probability',
        'prediction',
        'overlay',
        'bbox_xmin',
        'bbox_ymin',
        'bbox_xmax',
        'bbox_ymax',
        'bbox_xmin_norm',
        'bbox_ymin_norm',
        'bbox_xmax_norm',
        'bbox_ymax_norm',
        'bbox_area_pct',
        'bbox_mean_activation'
    ]
    
    # Sort rows by image number (increasing)
    selected_rows.sort(key=lambda r: int(os.path.splitext(r['image'])[0]))

    with open(NEW_PREDICTIONS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            # Only write keys that are in fieldnames
            filtered_row = {k: row.get(k, '') for k in fieldnames}
            writer.writerow(filtered_row)
            
    print(f"CSV saved to: {NEW_PREDICTIONS_CSV}")
    print("Done!")

if __name__ == "__main__":
    main()
