import os
import random
import csv
import shutil

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# New source structure
SOURCE_GRADCAM_BASE = os.path.join(ROOT_DIR, 'predicted')
SOURCE_ORIGINAL_BASE = os.path.join(ROOT_DIR, 'dataset', 'test')

# We will sample based on GradCAM folders
SOURCE_DIRS = {
    '0': os.path.join(SOURCE_GRADCAM_BASE, '0'),
    '2': os.path.join(SOURCE_GRADCAM_BASE, '2'),
    '3': os.path.join(SOURCE_GRADCAM_BASE, '3'),
    '4': os.path.join(SOURCE_GRADCAM_BASE, '4')
}

DEST_DIR = os.path.join(BASE_DIR, 'sampled')
DEST_MAP = os.path.join(DEST_DIR, 'map')
DEST_NO_MAP = os.path.join(DEST_DIR, 'no_map')

PREDICTIONS_CSV = os.path.join(SOURCE_GRADCAM_BASE, 'predictions.csv')
NEW_PREDICTIONS_CSV = os.path.join(DEST_DIR, 'predictions.csv')

def main():
    # Ensure destination directories exist
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_MAP)
    os.makedirs(DEST_NO_MAP)

    # Read predictions.csv to get metadata
    csv_data = {}
    fieldnames = []

    try:
        with open(PREDICTIONS_CSV, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            for row in reader:
                # Extract filename from path
                full_path = row['image']
                filename = os.path.basename(full_path)
                # Extract class from path (e.g. dataset/test/0/...)
                parts = full_path.split('/')
                if len(parts) >= 2:
                    # The class is the folder name before the filename
                    # e.g. dataset/test/0/file.png -> '0'
                    cls = parts[-2]
                    csv_data[(cls, filename)] = row
    except FileNotFoundError:
        print(f"Error: {PREDICTIONS_CSV} not found.")
        return

    # Collect available images
    available_images = {}
    for cls, path in SOURCE_DIRS.items():
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            available_images[cls] = images
        else:
            print(f"Warning: Directory {path} does not exist.")
            available_images[cls] = []

    # Validation
    if len(available_images['0']) < 25:
        print(f"Error: Not enough images in class 0. Found {len(available_images['0'])}, need 25.")
        return

    for cls in ['2', '3', '4']:
        if len(available_images[cls]) < 5:
            print(f"Error: Not enough images in class {cls}. Found {len(available_images[cls])}, need at least 5.")
            return

    selected_images = [] # List of tuples (class, filename, src_path)

    # Sample from Class 0
    sample_0 = random.sample(available_images['0'], 25)
    for img in sample_0:
        selected_images.append(('0', img, os.path.join(SOURCE_DIRS['0'], img)))

    # Sample from Classes 2, 3, 4
    pool_234 = []
    remaining_needed = 25
    
    # First, take 5 from each
    for cls in ['2', '3', '4']:
        s = random.sample(available_images[cls], 5)
        for img in s:
            selected_images.append((cls, img, os.path.join(SOURCE_DIRS[cls], img)))
        remaining_needed -= 5
        
        # Add rest to pool
        rest = [img for img in available_images[cls] if img not in s]
        for img in rest:
            pool_234.append((cls, img, os.path.join(SOURCE_DIRS[cls], img)))

    # Sample remaining from pool
    if len(pool_234) < remaining_needed:
        print(f"Error: Not enough remaining images in classes 2, 3, 4. Need {remaining_needed}, have {len(pool_234)}.")
        return

    sample_rest = random.sample(pool_234, remaining_needed)
    selected_images.extend(sample_rest)

    # Shuffle
    random.shuffle(selected_images)

    # Process
    new_csv_rows = []
    
    # Define new columns
    # User requested 'image_name_original' as well.
    # We keep 'image' (new filename), 'image_name' (original filename), 'image_name_original' (original filename), 'class'
    # And then all original fields except 'image' (which is replaced by the new filename)
    # Also adding 'ai_confidence'
    
    original_fields = [f for f in fieldnames if f != 'image']
    new_fieldnames = ['image', 'image_name', 'image_name_original', 'class', 'ai_confidence'] + original_fields

    print(f"Original fields found: {original_fields}")

    for idx, (cls, filename, src_path) in enumerate(selected_images, 1):
        new_filename = f"{idx}.png"
        
        # Copy GradCAM image (stored under 'map')
        dst_gradcam_path = os.path.join(DEST_MAP, new_filename)
        shutil.copy2(src_path, dst_gradcam_path)
        
        # Determine and copy Original image
        # GradCAM filename: X_gradcam.png -> Original: X.png
        original_filename = filename.replace('_gradcam', '')
        src_original_path = os.path.join(SOURCE_ORIGINAL_BASE, cls, original_filename)
        
        if os.path.exists(src_original_path):
            dst_original_path = os.path.join(DEST_NO_MAP, new_filename)
            shutil.copy2(src_original_path, dst_original_path)
        else:
            print(f"Warning: Original image not found for {filename} at {src_original_path}")

        # Try to match filename by stripping _gradcam if present
        lookup_filename = filename.replace('_gradcam', '')
        original_row = csv_data.get((cls, lookup_filename))
        
        # Fallback search
        if not original_row:
             # Try removing _gradcam suffix if present
             clean_name = filename.replace('_gradcam', '')
             original_row = csv_data.get((cls, clean_name))
             
             if not original_row:
                for k, v in csv_data.items():
                    if k[1] == filename or k[1] == clean_name:
                        original_row = v
                        break
        
        if not original_row:
            print(f"Warning: No metadata found for {filename} (Class {cls})")

        # Calculate AI Confidence
        ai_confidence = ''
        if original_row and 'probability' in original_row:
            try:
                prob = float(original_row['probability'])
                # Confidence = |Probability - 0.5| * 2
                conf = abs(prob - 0.5) * 2
                ai_confidence = f"{conf:.4f}"
            except ValueError:
                pass

        row_to_write = {
            'image': new_filename,
            'image_name': filename,
            'image_name_original': original_filename,
            'class': cls,
            'ai_confidence': ai_confidence
        }
        
        if original_row:
            for field in original_fields:
                row_to_write[field] = original_row.get(field, '')
        else:
            # Fill with empty
            for field in original_fields:
                row_to_write[field] = ''
                
        new_csv_rows.append(row_to_write)

    with open(NEW_PREDICTIONS_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(new_csv_rows)

    print(f"Success. Sampled {len(selected_images)} images to {DEST_DIR}")

if __name__ == "__main__":
    main()
