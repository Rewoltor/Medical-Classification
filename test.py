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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# --- Configuration ---
CONFIG = {
    'test_dir': "./dataset/test",
    'model_path': "./arthritis_classifier.pth",
    'output_dir': "./eval_results",
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
            # Handle corrupt images gracefully during inference
            print(f"Warning: Failed to load {img_path}. Skipping.")
            return torch.zeros((3, CONFIG['input_size'], CONFIG['input_size'])), torch.tensor(label, dtype=torch.float32)

# --- Model Loader ---
def load_model(device):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    
    # Match architecture from training
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

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = get_device()
    print(f"Running evaluation on {device}")

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

    print("Starting inference...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs))
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Calculate Null Accuracy (Baseline)
    values, counts = np.unique(y_true, return_counts=True)
    majority_class_idx = np.argmax(counts)
    null_accuracy = counts[majority_class_idx] / len(y_true)

    # Output Generation
    print("\n--- Performance Metrics ---")
    print(f"Accuracy:      {acc:.4f} (Baseline: {null_accuracy:.4f})")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    # Save Artifacts
    save_confusion_matrix(y_true, y_pred, CONFIG['output_dir'])
    print(f"Artifacts saved to {CONFIG['output_dir']}")