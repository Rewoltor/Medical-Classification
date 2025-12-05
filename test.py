import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Added imports for structure and logging
import csv
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_curve, auc

# --- Configuration ---
CONFIG = {
    'test_dir': "./dataset/test",
    'model_path': "./arthritis_classifier.pth",
    'output_dir': "./eval_results", # This acts as the root now (like ./eval in train.py)
    'batch_size': 32,
    'input_size': 224,
    'num_workers': 0,
    'class_map': {'0': 0, '2': 1, '3': 1, '4': 1} # 0: Negative, 1: Positive
}

# --- Device Setup ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class ArthritisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        for class_folder, label_val in CONFIG['class_map'].items():
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.exists(class_path):
                continue
                
            img_paths = glob.glob(os.path.join(class_path, "*.png"))
            self.file_paths.extend(img_paths)
            self.labels.extend([label_val] * len(img_paths))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}. Skipping.")
            return torch.zeros((3, CONFIG['input_size'], CONFIG['input_size'])), torch.tensor(label, dtype=torch.float32)

# --- Model Loader ---
def load_model(device):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )
    
    if os.path.exists(CONFIG['model_path']):
        state_dict = torch.load(CONFIG['model_path'], map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model weights not found at {CONFIG['model_path']}")
        
    return model.to(device)

# --- Visualization ---
def save_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_pr_auc_curve(y_true, y_probs, output_dir):
    # Precision-Recall curve and Average Precision (area under PR curve)
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.2f})')
    # Baseline: horizontal line at positive class prevalence
    positive_rate = np.mean(y_true) if len(y_true) > 0 else 0.0
    plt.hlines(positive_rate, 0, 1, colors='navy', linestyles='--', lw=2, label=f'Baseline = {positive_rate:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (AUC)')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_curve.png'))
    plt.close()
    return ap

def save_roc_curve(y_true, y_probs, output_dir):
    # Restore ROC plotting (area under ROC curve)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    return roc_auc

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Dynamic Directory Creation (From train.py) ---
    base_eval_dir = CONFIG['output_dir']
    os.makedirs(base_eval_dir, exist_ok=True)

    # Determine next numeric run folder
    existing = os.listdir(base_eval_dir)
    max_idx = 0
    for fn in existing:
        full = os.path.join(base_eval_dir, fn)
        if not os.path.isdir(full):
            continue
        try:
            idx = int(fn)
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue
    next_idx = max_idx + 1
    run_dir = os.path.join(base_eval_dir, str(next_idx))
    os.makedirs(run_dir, exist_ok=True)
    
    device = get_device()
    print(f"Running evaluation on {device}")
    print(f"Saving results to: {run_dir}")

    # Data Setup
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ArthritisDataset(CONFIG['test_dir'], transform=test_transform)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    print(f"Dataset loaded: {len(dataset)} samples")

    # Inference
    model = load_model(device)
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = []

    print("Starting inference...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs)
            preds = torch.round(probs)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy().flatten())

    # Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    values, counts = np.unique(y_true, return_counts=True)
    majority_class_idx = np.argmax(counts)
    null_accuracy = counts[majority_class_idx] / len(y_true)
    
    # Save artifacts to the specific run_dir
    roc_auc = save_roc_curve(y_true, y_probs, run_dir)
    pr_ap = save_pr_auc_curve(y_true, y_probs, run_dir)
    save_confusion_matrix(y_true, y_pred, run_dir)

    print("\n--- Performance Metrics ---")
    print(f"Accuracy:      {acc:.4f} (Baseline: {null_accuracy:.4f})")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"ROC AUC:       {roc_auc:.4f}")
    print(f"PR AUC (AP):   {pr_ap:.4f}")

    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    print(report)

    # --- 2. Save CSV Logs (From train.py) ---
    csv_path = os.path.join(run_dir, f"test_results.csv")
    
    # Metadata dictionary
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_path': CONFIG['model_path'],
        'test_dir': CONFIG['test_dir'],
        'device': str(device),
        'dataset_size': len(dataset)
    }

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write Metadata
        writer.writerow(['metadata_key', 'metadata_value'])
        for k, v in metadata.items():
            writer.writerow([k, v])
        writer.writerow([])
        
        # Write Metrics
        writer.writerow(['metric', 'value'])
        writer.writerow(['Accuracy', acc])
        writer.writerow(['Null_Accuracy', null_accuracy])
        writer.writerow(['Precision', precision])
        writer.writerow(['Recall', recall])
        writer.writerow(['F1_Score', f1])
        writer.writerow(['ROC_AUC', roc_auc])
        writer.writerow(['PR_AUC', pr_ap])
    
    print(f"Evaluation CSV saved to {csv_path}")