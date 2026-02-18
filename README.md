# Medical Classifier - Percotate Research Project

This repository contains the codebase for the "Medical Classifier" research project. It focuses on the automated severity grading of Knee Osteoarthritis (KOA) using deep learning (ResNet-18) and rigorous evaluation methodologies for Human-AI collaboration studies.

**Audience:** Researchers and Reviewers

## 1. Dataset

This project utilizes the **Knee Osteoarthritis Severity Grading Dataset**:

*   **Source:** [Mendeley Data](https://data.mendeley.com/datasets/56rmx5bjcr/1)
*   **Citation:** Chen, P. (2018). Knee osteoarthritis severity grading dataset. Mendeley Data, V1. doi: 10.17632/56rmx5bjcr.1
*   **Structure:**
    *   The dataset is organized by KL (Kellgren-Lawrence) grades (0-4).
    *   `0`: Healthy
    *   `1`: Doubtful
    *   `2`: Minimal
    *   `3`: Moderate
    *   `4`: Severe

## 2. Installation

### Prerequisites
*   Python 3.8+
*   pip

### Setup

1.  **Clone the repository** & navigate to the project root.
2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Environment Variables:**
    Create a `.env` file in the root directory if specific environment configurations are needed (refer to `.env.example` if available).

## 3. Workflows

### A. Training the Model (`train.py`)
Trains a ResNet-18 model on the dataset. The script handles data augmentation, class balancing, and fine-tuning.

*   **Command:** `python3 train.py`
*   **Output:** Saves the best model weights to `arthritis_classifier_best.pth` and logs training history.

### B. Evaluating Performance (`test.py`)
Evaluates the trained model on the test set. Generates comprehensive metrics including Confusion Matrix, ROC-AUC, and PR-AUC curves.

*   **Command:** `python3 test.py`
*   **Output:** Results are saved in the `eval_results/` directory.

### C. Running Inference & GradCAM (`predict.py`)
Runs inference on test images and generates GradCAM (Gradient-weighted Class Activation Mapping) overlays to visualize model attention.

*   **Command:** `python3 predict.py`
*   **Output:**
    *   Generates `predicted/predictions.csv` containing raw logits, probabilities, and bounding box data.
    *   Saves images with GradCAM overlays in `predicted/`.

### D. Rigorous Sampling for Human Studies (`randomSample/`)

Scripts to generate scientifically rigorous subsets of data for Human-AI interaction experiments.

#### 1. Stratified Confusion Matrix Sampling (`randomSample.py`)
Implements the "DANNY" methodology (Jeon et al., 2025) to create a balanced dataset representing the AI's confusion matrix (TP, TN, FP, FN).

*   **Command:** `python3 randomSample/randomSample.py`
*   **Output:** Creates `randomSample/sampled/` with 50 images balanced across error types.

#### 2. Random 10 Sampling (`randomSample/random 5.py`)
Extracts a purely random set of 10 images (2 from each KL grade 0-4) for quick validation or specific small-scale tests.

*   **Command:** `python3 "randomSample/random 5.py"`
*   **Output:**
    *   Creates `randomSample/sampled_5/`.
    *   Images are renamed `51.png` - `60.png`.
    *   Includes `predictions.csv` with full metadata for the selected images.

## 4. Project Structure

*   `dataset/`: Contains the specific training and testing data splits.
*   `randomSample/`: Sampling logic for experiment generation.
*   `eval_results/`: Output directory for evaluation metrics and graphs.
*   `predicted/`: Output directory for inference results and GradCAM visualizations.
*   `arthritis_classifier.pth`: Trained model weights.

---
**Contact:** For questions regarding this implementation or the associated research, please contact the repository maintainer.
