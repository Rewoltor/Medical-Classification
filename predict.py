import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import glob
import csv
from pathlib import Path

# --- Configuration ---
MODEL_PATH = "./arthritis_classifier.pth"
TEST_DIR = "./dataset/test"
OUTPUT_DIR = "./predicted"
COMMON_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
INPUT_SIZE = 224

def collect_images(test_dir):
    imgs = []
    for root, _, _ in os.walk(test_dir):
        for ext in COMMON_EXTS:
            imgs.extend(glob.glob(os.path.join(root, ext)))
    return sorted(imgs)

# device
if not torch.backends.mps.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("mps")

# --- Model ---
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
# Match the training model: Dropout followed by a Linear layer.
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- Hooks ---
gradients = None
activations = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    # grad_output is a tuple
    gradients = grad_output[0].detach()

# Use register_full_backward_hook if available (preferred)
target_layer = model.layer4
fwd_handle = target_layer.register_forward_hook(forward_hook)
try:
    bwd_handle = target_layer.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
except Exception:
    # Fallback for older PyTorch
    bwd_handle = target_layer.register_backward_hook(backward_hook)

# --- Preprocess ---
preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

os.makedirs(OUTPUT_DIR, exist_ok=True)
pred_csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")

image_paths = collect_images(TEST_DIR)
if not image_paths:
    print(f"No images found under {TEST_DIR}. Exiting.")
    raise SystemExit(1)

results = []

for img_path in image_paths:
    # reset captured tensors
    gradients = None
    activations = None

    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open {img_path}: {e}. Skipping.")
        continue

    # original image size (width, height) — we'll report bbox in original coords
    orig_w, orig_h = pil_img.size

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    model.zero_grad()
    with torch.enable_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred_class = 1 if prob > 0.5 else 0
        # backward
        output.backward(torch.ones_like(output))

    recorded_heatmap = None
    if gradients is None or activations is None:
        print(f"Grad-CAM hooks failed for {img_path}; saving original image without overlay.")
    else:
        # pooled gradients
        if gradients.ndim == 4:
            pooled = torch.mean(gradients, dim=(0, 2, 3))
        else:
            pooled = torch.mean(gradients, dim=0)

        act = activations.clone()
        for i in range(act.shape[1]):
            act[:, i, :, :] *= pooled[i]

        heatmap = torch.mean(act, dim=1).squeeze()
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = np.maximum(heatmap_np, 0)
        maxv = np.max(heatmap_np) if heatmap_np.size else 0.0
        if maxv > 0:
            heatmap_np = heatmap_np / maxv
        else:
            heatmap_np = np.zeros_like(heatmap_np)
        recorded_heatmap = heatmap_np

    # prepare output path
    rel_dir = os.path.relpath(os.path.dirname(img_path), TEST_DIR)
    out_subdir = os.path.join(OUTPUT_DIR, rel_dir) if rel_dir != "." else OUTPUT_DIR
    os.makedirs(out_subdir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_image_path = os.path.join(out_subdir, f"{base_name}_gradcam.png")

    # create overlay and compute bbox from heatmap (in original image coords)
    # default bbox values
    bbox_abs = ("", "", "", "")
    bbox_norm = ("", "", "", "")
    bbox_area_pct = ""
    bbox_mean_activation = ""

    try:
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # keep original resolution for overlay and bbox
        img_h, img_w = img_cv.shape[:2]

        if recorded_heatmap is not None:
            # resize heatmap to original image resolution
            heatmap_resized = cv2.resize(recorded_heatmap, (img_w, img_h))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)

            # threshold to produce mask of high-activation areas
            thresh_val = 0.2  # relative threshold (tunable)
            _, mask = cv2.threshold(np.uint8(255 * heatmap_resized), int(255 * thresh_val), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)
                bbox_abs = (xmin, ymin, xmax, ymax)
                bbox_norm = (xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h)
                bbox_area_pct = 100.0 * (w * h) / (img_w * img_h)
                mean_act = float(np.mean(heatmap_resized[y:y+h, x:x+w])) if (w*h) > 0 else 0.0
                bbox_mean_activation = mean_act

            # generate color heatmap and overlay
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
        else:
            superimposed = img_cv

        # NOTE: intentionally do NOT draw the bbox on the overlay image.
        # We still compute and export bbox coordinates to the CSV, but
        # the visual overlay should remain heatmap-only per user request.

        cv2.imwrite(out_image_path, superimposed)
    except Exception as e:
        print(f"Failed to create/save overlay for {img_path}: {e}")

    results.append({
        "image": os.path.relpath(img_path),
        "output_logit": float(output.item()),
        "probability": float(prob),
        "prediction": int(pred_class),
        "overlay": os.path.relpath(out_image_path),
        "bbox_xmin": bbox_abs[0] if bbox_abs[0] != "" else "",
        "bbox_ymin": bbox_abs[1] if bbox_abs[0] != "" else "",
        "bbox_xmax": bbox_abs[2] if bbox_abs[0] != "" else "",
        "bbox_ymax": bbox_abs[3] if bbox_abs[0] != "" else "",
        "bbox_xmin_norm": f"{bbox_norm[0]:.6f}" if bbox_abs[0] != "" else "",
        "bbox_ymin_norm": f"{bbox_norm[1]:.6f}" if bbox_abs[0] != "" else "",
        "bbox_xmax_norm": f"{bbox_norm[2]:.6f}" if bbox_abs[0] != "" else "",
        "bbox_ymax_norm": f"{bbox_norm[3]:.6f}" if bbox_abs[0] != "" else "",
        "bbox_area_pct": float(f"{bbox_area_pct:.4f}") if bbox_area_pct != "" else "",
        "bbox_mean_activation": float(f"{bbox_mean_activation:.6f}") if bbox_mean_activation != "" else ""
    })

    print(f"Processed: {img_path} -> pred={pred_class} prob={prob:.4f} overlay={out_image_path}")

# write CSV
with open(pred_csv_path, "w", newline="") as csvfile:
    fieldnames = [
        "image", "output_logit", "probability", "prediction", "overlay",
        "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        "bbox_xmin_norm", "bbox_ymin_norm", "bbox_xmax_norm", "bbox_ymax_norm",
        "bbox_area_pct", "bbox_mean_activation"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"\nDone. Overlays and predictions.csv written to: {OUTPUT_DIR}")

# cleanup hooks
fwd_handle.remove()
try:
    bwd_handle.remove()
except Exception:
    pass