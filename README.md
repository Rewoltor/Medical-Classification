# Medical-Classification

A small PyTorch project for binary classification of arthritis images using a fine-tuned ResNet-18.
The repository includes training, evaluation and prediction (Grad-CAM overlays) scripts.

## Contents
- `train.py` — training / fine-tuning script (expects `dataset/train` and `dataset/val`)
- `test.py` — evaluation script (expects `dataset/test`)
- `predict.py` — batch inference + Grad-CAM overlays (reads images under `dataset/test` and writes `predicted/`)
- `best_arthritis_classifier.pth` — example checkpoint filename (keep model files out of Git)
- `.gitignore` — excludes venv, dataset, predictions and model files

## Quick facts
- Input size: 224x224 (ResNet-18)
- Loss: `BCEWithLogitsLoss` (binary output)
- Label mapping (used by the dataset classes):
  - folder `0` -> label `0` (negative)
  - folders `2`, `3`, `4` -> label `1` (positive)
  - folder `1` is ignored by the training script if present
- `train.py` and `test.py` currently load `*.png` images; `predict.py` accepts PNG/JPG/JPEG/BMP

## Expected dataset layout
Place your images under a `dataset/` folder with the following structure:

```
dataset/
  train/
    0/    # png images for class 0
    2/
    3/
    4/
  val/
    0/
    2/
    3/
    4/
  test/
    0/    # or arbitrary nested images for predict.py
    2/
    3/
    4/
```

Notes:
- If your images are JPEG, either convert them to PNG or update `train.py`/`test.py` to accept multiple extensions.
- `predict.py` will search recursively under `dataset/test` and write overlays to `predicted/` plus `predicted/predictions.csv`.

## Setup (macOS, M1 / Apple Silicon recommended)
Run these from the project root.

1. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install lightweight deps
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install pillow numpy scikit-learn opencv-python six
```

3. Install PyTorch / torchvision (choose MPS for Apple Silicon)
- Visit https://pytorch.org/get-started/locally/ and select the appropriate options (pip, macOS, MPS) and run the suggested command.
- Example (verify on site before running):
```bash
# Example: CPU or MPS specific wheel from the official installer
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

4. Verify installation
```bash
python -c "import torch, torchvision, json; print(json.dumps({'torch':torch.__version__, 'torchvision':torchvision.__version__, 'mps':torch.backends.mps.is_available()}))"
```

## Run
- Train (requires `dataset/train` and `dataset/val`):
```bash
source .venv/bin/activate
python train.py
```
- Test (uses `dataset/test` and `best_arthritis_classifier.pth`):
```bash
python test.py
```
- Predict + Grad-CAM overlays:
```bash
python predict.py
```

## Git / Large files
- `.gitignore` already excludes `.venv/`, `dataset/`, `predicted/`, and `arthritis_classifier.pth`.
- If any of these are currently tracked, untrack them (keeps copies on disk but removes from git index):
```bash
git rm -r --cached .venv dataset predicted arthritis_classifier.pth || true
git add .gitignore
git commit -m "Stop tracking venv, dataset, predicted and model file" || true
git push origin main
```

If large files were pushed to remote history earlier, let me know and I will help you remove them with `git-filter-repo` or BFG (this rewrites history and requires a force push).

## Troubleshooting
- `ModuleNotFoundError` for small packages like `six`: install them in the venv (`python -m pip install six`).
- If torchvision raises import errors after upgrades, reinstall a matching torchvision for your installed torch (`pip uninstall torchvision && pip install torchvision`).
- If OpenCV/Numpy conflicts appear, pin numpy to a compatible version (`python -m pip install "numpy<2.3.0"`).

## Next steps / optional improvements
- Generate `requirements.txt`: `python -m pip freeze > requirements.txt`
- Accept multiple image extensions in `train.py`/`test.py` (I can update the scripts).
- Add a small dataset summary script to print counts per class/split.

If you'd like any of the optional improvements, pick one and I will implement it.
