#!/usr/bin/env python3
"""
Stratified Confusion Matrix Sampling Script
============================================

Q1 Journal Submission Version - Maximum Scientific Rigor

This script implements the "DANNY" paper (Jeon et al., 2025) methodology
for constructing a rigorous dataset for Human-AI collaboration studies
on Knee Osteoarthritis detection.

Key Design Decisions:
--------------------
1. STRICT DANNY RATIOS: 7 FP + 18 TN (Healthy), 8 FN + 17 TP (Arthritis)
2. PURE RANDOM SAMPLING: No manufactured difficulty curves - we test the
   AI's natural error distribution to avoid selection bias.
3. SCIENTIFIC HONESTY: Confidence metric labeled as 'softmax_confidence'
   (proxy metric, not Entropy as in original paper).
4. REPRODUCIBILITY: Deterministic with random.seed(42).

Author: Publication-ready version for Q1 journal submission
"""

import os
import random
import csv
import shutil
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Reproducibility: Deterministic dataset generation
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Source directories
PREDICTIONS_CSV = os.path.join(ROOT_DIR, 'predicted', 'predictions.csv')
SOURCE_GRADCAM_BASE = os.path.join(ROOT_DIR, 'predicted')
SOURCE_ORIGINAL_BASE = os.path.join(ROOT_DIR, 'dataset', 'test')

# Destination directories
DEST_DIR = os.path.join(BASE_DIR, 'sampled')
DEST_MAP = os.path.join(DEST_DIR, 'map')
DEST_NO_MAP = os.path.join(DEST_DIR, 'no_map')
NEW_PREDICTIONS_CSV = os.path.join(DEST_DIR, 'predictions.csv')

# Quality Control: Add filenames here to exclude broken/invalid images
# Example: BLACKLIST = ['9003175L.png', '9003316L.png']
BLACKLIST: List[str] = []

# =============================================================================
# SAMPLING TARGETS (DANNY Methodology - Jeon et al., 2025)
# =============================================================================

# Total images = 50
TOTAL_HEALTHY = 25
TOTAL_ARTHRITIS = 25

# Group A: Healthy (ground_truth_raw == 0)
TARGET_FP = 7    # False Positives: AI says arthritis, but actually healthy
TARGET_TN = 18   # True Negatives: AI correctly says healthy

# Group B: Arthritis (ground_truth_raw in {2, 3, 4})
TARGET_FN = 8    # False Negatives: AI misses arthritis
TARGET_TP = 17   # True Positives: AI correctly identifies arthritis

# Validate targets
assert TARGET_FP + TARGET_TN == TOTAL_HEALTHY, "Healthy targets must sum to 25"
assert TARGET_FN + TARGET_TP == TOTAL_ARTHRITIS, "Arthritis targets must sum to 25"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ImageRecord:
    """Represents a single image with its metadata from predictions.csv"""
    def __init__(self, row: Dict[str, str]):
        self.row = row
        self.image_path = row['image']  # e.g., dataset/test/0/9003175L.png
        self.ground_truth_raw = int(row['ground_truth_raw'])
        self.prediction = int(row['prediction'])
        self.probability = float(row['probability'])
        
        # Extract filename and class from path
        parts = self.image_path.split('/')
        self.filename = parts[-1]  # e.g., 9003175L.png
        self.kl_grade = parts[-2]  # e.g., '0'
        
        # GradCAM path: Use 'overlay' column if available, else construct manually
        if 'overlay' in row and row['overlay'].strip():
            self.gradcam_path = os.path.join(ROOT_DIR, row['overlay'])
            self.gradcam_filename = os.path.basename(row['overlay'])
        else:
            name_without_ext = os.path.splitext(self.filename)[0]
            self.gradcam_filename = f"{name_without_ext}_gradcam.png"
            self.gradcam_path = os.path.join(SOURCE_GRADCAM_BASE, self.kl_grade, self.gradcam_filename)
        
        # Original path: dataset/test/CLASS/FILENAME
        self.original_path = os.path.join(SOURCE_ORIGINAL_BASE, self.kl_grade, self.filename)
    
    @property
    def is_healthy(self) -> bool:
        return self.ground_truth_raw == 0
    
    @property
    def is_arthritis(self) -> bool:
        return self.ground_truth_raw in {2, 3, 4}
    
    @property
    def softmax_confidence(self) -> float:
        """
        Softmax-derived confidence proxy: |probability - 0.5| * 2
        
        Note: This is NOT the Entropy metric used in Jeon et al. (2025).
        We acknowledge this limitation in our methodology.
        """
        return abs(self.probability - 0.5) * 2
    
    def get_confusion_category(self) -> Optional[str]:
        """
        Determine confusion matrix category:
        - TN: Healthy + AI predicts Healthy
        - FP: Healthy + AI predicts Arthritis
        - TP: Arthritis + AI predicts Arthritis
        - FN: Arthritis + AI predicts Healthy
        """
        if self.ground_truth_raw == 1:
            return None  # Excluded (ambiguous)
        
        if self.is_healthy:
            return 'FP' if self.prediction == 1 else 'TN'
        elif self.is_arthritis:
            return 'TP' if self.prediction == 1 else 'FN'
        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_predictions() -> List[ImageRecord]:
    """Load and filter predictions.csv, excluding KL Grade 1 and blacklisted images"""
    records = []
    
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_CSV}")
    
    with open(PREDICTIONS_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kl_grade = int(row['ground_truth_raw'])
            
            # EXCLUDE KL Grade 1 (ambiguous, invalid for study)
            if kl_grade == 1:
                continue
            
            # Quality Control: Skip blacklisted images
            filename = os.path.basename(row['image'])
            if filename in BLACKLIST:
                print(f"⚠️  Skipping blacklisted image: {filename}")
                continue
            
            records.append(ImageRecord(row))
    
    return records


def build_pools(records: List[ImageRecord]) -> Dict[str, List[ImageRecord]]:
    """
    Build the 4 confusion matrix pools.
    
    Returns dict with keys: 'TN', 'FP', 'TP', 'FN'
    Each pool contains all qualifying images for pure random sampling.
    """
    pools = defaultdict(list)
    
    for record in records:
        category = record.get_confusion_category()
        if category:
            pools[category].append(record)
    
    return pools


def sample_with_warning(pool: List[ImageRecord], target: int, pool_name: str) -> List[ImageRecord]:
    """
    Pure random sample from pool.
    Warns if insufficient images available.
    """
    if len(pool) < target:
        print(f"⚠️  WARNING: Pool '{pool_name}' has only {len(pool)} images, target was {target}. Using all available.")
        return pool.copy()
    return random.sample(pool, target)


def get_kl_breakdown(records: List[ImageRecord]) -> str:
    """Generate KL Grade breakdown string for reporting"""
    counts = Counter(r.ground_truth_raw for r in records)
    parts = []
    for grade in sorted(counts.keys()):
        parts.append(f"{counts[grade]}×KL{grade}")
    return ", ".join(parts) if parts else "none"


def stratified_sample(pools: Dict[str, List[ImageRecord]]) -> List[ImageRecord]:
    """
    Perform DANNY stratified sampling with PURE RANDOM selection.
    
    No artificial difficulty curves - we sample randomly within each
    confusion matrix bucket to test the AI's natural error distribution.
    
    Group A (Healthy): 7 FP + 18 TN = 25
    Group B (Arthritis): 8 FN + 17 TP = 25
    """
    selected = []
    
    # =========================================================================
    # GROUP A: HEALTHY (25 images)
    # =========================================================================
    
    print("\n" + "="*60)
    print("GROUP A: HEALTHY (Target: 25 images)")
    print("="*60)
    
    tn_pool = pools.get('TN', [])
    fp_pool = pools.get('FP', [])
    
    print(f"\n📊 Available TN (AI Correct): {len(tn_pool)}")
    print(f"📊 Available FP (AI Error): {len(fp_pool)}")
    
    # Pure random sampling from each pool
    fp_selected = sample_with_warning(fp_pool, TARGET_FP, "FP")
    tn_selected = sample_with_warning(tn_pool, TARGET_TN, "TN")
    
    selected.extend(fp_selected)
    selected.extend(tn_selected)
    
    print(f"\n✓ Selected FP: {len(fp_selected)} (target: {TARGET_FP})")
    print(f"✓ Selected TN: {len(tn_selected)} (target: {TARGET_TN})")
    print(f"✓ Total Healthy: {len(fp_selected) + len(tn_selected)}")
    
    # =========================================================================
    # GROUP B: ARTHRITIS (25 images)
    # =========================================================================
    
    print("\n" + "="*60)
    print("GROUP B: ARTHRITIS (Target: 25 images)")
    print("="*60)
    
    fn_pool = pools.get('FN', [])
    tp_pool = pools.get('TP', [])
    
    print(f"\n📊 Available FN (AI Error): {len(fn_pool)}")
    print(f"📊 Available TP (AI Correct): {len(tp_pool)}")
    
    # Pure random sampling - no manufactured difficulty curves
    fn_selected = sample_with_warning(fn_pool, TARGET_FN, "FN")
    tp_selected = sample_with_warning(tp_pool, TARGET_TP, "TP")
    
    selected.extend(fn_selected)
    selected.extend(tp_selected)
    
    # Report KL Grade breakdown for variance verification
    print(f"\n✓ Selected FN: {len(fn_selected)} (target: {TARGET_FN})")
    print(f"   └─ KL Breakdown: {get_kl_breakdown(fn_selected)}")
    print(f"✓ Selected TP: {len(tp_selected)} (target: {TARGET_TP})")
    print(f"   └─ KL Breakdown: {get_kl_breakdown(tp_selected)}")
    print(f"✓ Total Arthritis: {len(fn_selected) + len(tp_selected)}")
    
    return selected


def copy_images(selected: List[ImageRecord]) -> List[Tuple[int, ImageRecord]]:
    """Copy selected images to destination folders."""
    # Shuffle for randomized presentation order
    random.shuffle(selected)
    
    indexed_records = []
    
    for idx, record in enumerate(selected, 1):
        new_filename = f"{idx}.png"
        
        # Copy GradCAM (map) version
        gradcam_dest = os.path.join(DEST_MAP, new_filename)
        if os.path.exists(record.gradcam_path):
            shutil.copy2(record.gradcam_path, gradcam_dest)
        else:
            print(f"⚠️  WARNING: GradCAM not found: {record.gradcam_path}")
        
        # Copy Original (no_map) version
        original_dest = os.path.join(DEST_NO_MAP, new_filename)
        if os.path.exists(record.original_path):
            shutil.copy2(record.original_path, original_dest)
        else:
            print(f"⚠️  WARNING: Original not found: {record.original_path}")
        
        indexed_records.append((idx, record))
    
    return indexed_records


def generate_csv(indexed_records: List[Tuple[int, ImageRecord]]):
    """
    Generate sampled/predictions.csv preserving all original columns from source CSV.
    
    Columns:
    - image: new filename (1.png, 2.png, ...)
    - image_name: original GradCAM filename
    - image_name_original: original filename (without _gradcam)
    - class: KL Grade folder
    - ai_confidence: softmax-derived confidence proxy
    - All original columns from predictions.csv
    """
    # Define output fieldnames - matching the original format
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
    
    with open(NEW_PREDICTIONS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, record in indexed_records:
            # Build row with new columns + all original data
            row_to_write = {
                'image': f"{idx}.png",
                'image_name': record.gradcam_filename,
                'image_name_original': record.filename,
                'class': record.kl_grade,
                'ai_confidence': f"{record.softmax_confidence:.4f}",
            }
            
            # Copy all original columns from source row
            original_columns = [
                'ground_truth_raw', 'ground_truth_binary', 'output_logit',
                'probability', 'prediction', 'overlay',
                'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
                'bbox_xmin_norm', 'bbox_ymin_norm', 'bbox_xmax_norm', 'bbox_ymax_norm',
                'bbox_area_pct', 'bbox_mean_activation'
            ]
            
            for col in original_columns:
                row_to_write[col] = record.row.get(col, '')
            
            writer.writerow(row_to_write)


def print_summary(indexed_records: List[Tuple[int, ImageRecord]]):
    """Print final summary with KL Grade breakdown for variance verification"""
    print("\n" + "="*60)
    print("FINAL SUMMARY & VALIDATION")
    print("="*60)
    
    # Separate by confusion matrix category
    tn_records = [r for _, r in indexed_records if r.is_healthy and r.prediction == 0]
    fp_records = [r for _, r in indexed_records if r.is_healthy and r.prediction == 1]
    fn_records = [r for _, r in indexed_records if r.is_arthritis and r.prediction == 0]
    tp_records = [r for _, r in indexed_records if r.is_arthritis and r.prediction == 1]
    
    print(f"\nTotal Images: {len(indexed_records)}")
    print(f"Random Seed: 42 (deterministic)")
    print(f"Sampling Method: Pure Random (no artificial difficulty curves)")
    
    print(f"\n{'─'*60}")
    print("GROUP A - HEALTHY")
    print(f"{'─'*60}")
    print(f"  TN (AI Correct): {len(tn_records)} (target: {TARGET_TN})")
    print(f"  FP (AI Error):   {len(fp_records)} (target: {TARGET_FP})")
    print(f"  Total: {len(tn_records) + len(fp_records)}")
    
    print(f"\n{'─'*60}")
    print("GROUP B - ARTHRITIS (with KL Grade Breakdown)")
    print(f"{'─'*60}")
    print(f"  FN (AI Error):   {len(fn_records)} (target: {TARGET_FN})")
    print(f"     └─ Breakdown: {get_kl_breakdown(fn_records)}")
    print(f"  TP (AI Correct): {len(tp_records)} (target: {TARGET_TP})")
    print(f"     └─ Breakdown: {get_kl_breakdown(tp_records)}")
    print(f"  Total: {len(fn_records) + len(tp_records)}")
    
    # Validation
    print(f"\n{'─'*60}")
    all_valid = True
    
    if len(tn_records) + len(fp_records) != TOTAL_HEALTHY:
        print(f"❌ VALIDATION FAILED: Healthy count is {len(tn_records) + len(fp_records)}, expected {TOTAL_HEALTHY}")
        all_valid = False
    
    if len(fn_records) + len(tp_records) != TOTAL_ARTHRITIS:
        print(f"❌ VALIDATION FAILED: Arthritis count is {len(fn_records) + len(tp_records)}, expected {TOTAL_ARTHRITIS}")
        all_valid = False
    
    if all_valid:
        print("✅ VALIDATION PASSED: All target counts achieved!")
    
    print(f"\n📁 Output: {DEST_DIR}")
    print(f"📄 CSV: {NEW_PREDICTIONS_CSV}")
    
    if BLACKLIST:
        print(f"🚫 Blacklisted: {len(BLACKLIST)} images")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*60)
    print("STRATIFIED CONFUSION MATRIX SAMPLING")
    print("DANNY Methodology (Jeon et al., 2025)")
    print("Q1 Journal Submission | Pure Random | Seed: 42")
    print("="*60)
    
    # Validate source paths
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"❌ ERROR: Predictions CSV not found: {PREDICTIONS_CSV}")
        sys.exit(1)
    
    if not os.path.exists(SOURCE_GRADCAM_BASE):
        print(f"❌ ERROR: GradCAM directory not found: {SOURCE_GRADCAM_BASE}")
        sys.exit(1)
    
    if not os.path.exists(SOURCE_ORIGINAL_BASE):
        print(f"❌ ERROR: Original image directory not found: {SOURCE_ORIGINAL_BASE}")
        sys.exit(1)
    
    # Clean and create destination directories
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_MAP)
    os.makedirs(DEST_NO_MAP)
    
    # Step 1: Load predictions
    print("\n📖 Loading predictions.csv (excluding KL Grade 1)...")
    records = load_predictions()
    print(f"   Loaded {len(records)} valid records")
    if BLACKLIST:
        print(f"   Blacklist: {len(BLACKLIST)} images excluded")
    
    # Step 2: Build confusion matrix pools
    print("\n🗂️  Building confusion matrix pools...")
    pools = build_pools(records)
    
    # Step 3: Perform pure random sampling
    print("\n🎯 Performing pure random sampling (no difficulty curves)...")
    selected = stratified_sample(pools)
    
    # Step 4: Copy images
    print("\n📋 Copying images...")
    indexed_records = copy_images(selected)
    print(f"   Copied {len(indexed_records)} images")
    
    # Step 5: Generate CSV
    print("\n📝 Generating predictions.csv...")
    generate_csv(indexed_records)
    
    # Step 6: Summary
    print_summary(indexed_records)


if __name__ == "__main__":
    main()
