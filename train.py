import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import datetime

# Configuration from previous paper
LEARNING_RATE_NEW_LAYER = 0.001
LEARNING_RATE_FINE_TUNE = 0.0001   
BATCH_SIZE = 32
NUM_EPOCHS = 5                 # maybe 20-30 epocs on few images
INPUT_SIZE = 224
TRAIN_LIMIT = 400              # max training samples used (set None to disable) -› number of images used to train
NUM_WORKERS = 0
RANDOM_SEED = 42               # seed for reproducible shuffling

TRAIN_DIR = "./dataset/train"
VAL_DIR = "./dataset/val"
MODEL_SAVE_PATH = "./arthritis_classifier.pth" 
EVAL_DIR = "./eval"

# --- 2. Set up the 'mps' Device ---

if not torch.backends.mps.is_available():
    print("MPS not available. Using CPU.")
    device = torch.device("cpu")
else:
    print("MPS is available. Using M1 GPU.")
    device = torch.device("mps")

# Set random seeds for reproducibility (where applicable)
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

# --- 3. Custom Dataset Class (Modified to limit training data) ---

# --- 3. Custom Dataset Class (MODIFIED FOR STRATIFIED SAMPLING) ---

class ArthritisDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        # We will apply the limit only if 'train' is in the root_dir
        is_train_set = "train" in root_dir and limit is not None
        
        # We want a balanced set, so we'll take limit / 2 from each class
        limit_per_class = int(limit / 2) if is_train_set else None

        label_map = {'0': 0, '2': 1, '3': 1, '4': 1}
        class_counts = {0: 0, 1: 0}
        
        print(f"Loading data from: {root_dir}")
        
        for class_folder in os.listdir(root_dir):
            if class_folder in label_map:
                label = label_map[class_folder]
                class_path = os.path.join(root_dir, class_folder)
                
                # Get all images for this class
                images_in_class = [
                    os.path.join(class_path, img) 
                    for img in os.listdir(class_path) 
                    if img.endswith(".png")
                ]
                
                # --- STRATIFIED SAMPLING LOGIC ---
                # If this is the training set, shuffle and limit this class
                if is_train_set:
                    random.shuffle(images_in_class)
                    if limit_per_class is not None:
                         images_in_class = images_in_class[:limit_per_class]
                # --- END OF SAMPLING LOGIC ---
                
                # Add the selected images to our dataset
                for img_path in images_in_class:
                    self.file_paths.append(img_path)
                    self.labels.append(label)
                    class_counts[label] += 1
                    
            elif class_folder == '1':
                # Ignoring Grade 1 as per the paper [cite: 239]
                pass
            else:
                print(f"Warning: Found unexpected folder, ignoring: {class_folder}")

        # Now, shuffle the *entire* final dataset (paths and labels together)
        temp_list = list(zip(self.file_paths, self.labels))
        random.shuffle(temp_list)
        self.file_paths, self.labels = zip(*temp_list)
        
        print(f"  -> Loaded {len(self.file_paths)} images.")
        print(f"  -> Label counts: {class_counts}")
        if is_train_set and (class_counts[0] != class_counts[1]):
             print(f"  -> WARNING: Training set is imbalanced. Check image counts.")
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

# --- 4. Define Data Transforms and Create DataLoaders ---

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15), # <-- Increased from 10
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([ 
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = ArthritisDataset(root_dir=TRAIN_DIR, transform=data_transforms['train'], limit=TRAIN_LIMIT)
val_dataset = ArthritisDataset(root_dir=VAL_DIR, transform=data_transforms['val'], limit=None)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) 

print(f"Data loaded: {len(train_dataset)} train images, {len(val_dataset)} validation images.") 

# --- 5. Define the ResNet-18 Model and Fine-Tuning Setup ---

model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last convolutional block (model.layer4)
for param in model.layer4.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features

# Replace the final layer with a sequence including Dropout
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # <-- ADD DROPOUT (50% probability)
    nn.Linear(num_ftrs, 1)
)

model = model.to(device)

# --- 6. Define Loss Function and Optimizer ---

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW([
    {'params': model.fc.parameters(), 'lr': LEARNING_RATE_NEW_LAYER},
    {'params': model.layer4.parameters(), 'lr': LEARNING_RATE_FINE_TUNE},
], weight_decay=1e-4)

# --- 7. Training and Validation Loop ---

best_val_loss = float('inf')

# History for training visualization
history = {
    'loss': [],
    'val_loss': [],
    'val_acc': [],
    'train_acc': []
}

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()  
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1) 
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        # compute training accuracy for this batch
        preds = torch.round(torch.sigmoid(outputs))
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_train_acc = running_corrects.float() / len(train_dataset)
    
    # --- Validation Phase ---
    model.eval()  
    val_loss = 0.0
    val_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1) 
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            preds = torch.round(torch.sigmoid(outputs))
            val_corrects += torch.sum(preds == labels.data)
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.float() / len(val_dataset) 
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    # record history for visualization
    history['loss'].append(epoch_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc.item())
    history['train_acc'].append(epoch_train_acc.item())
    
    # --- Save the Best Model ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> Validation loss improved. Model saved to {MODEL_SAVE_PATH}")

print("Finished training.")

# --- Evaluation visualization and saving to eval/ ---
try:
    eval_dir = EVAL_DIR
    os.makedirs(eval_dir, exist_ok=True)

    # determine next numeric run folder (look for numeric subdirectories)
    existing = os.listdir(eval_dir)
    max_idx = 0
    for fn in existing:
        full = os.path.join(eval_dir, fn)
        if not os.path.isdir(full):
            continue
        try:
            idx = int(fn)
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue
    next_idx = max_idx + 1
    run_dir = os.path.join(eval_dir, str(next_idx))
    os.makedirs(run_dir, exist_ok=True)

    # Create a combined figure: loss and accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=160)
    ax1.plot(history['loss'], label='train')
    ax1.plot(history['val_loss'], label='val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='train_acc')
    ax2.plot(history['val_acc'], label='val_acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    # Metadata to display and save
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'lr_new_layer': LEARNING_RATE_NEW_LAYER,
        'lr_fine_tune': LEARNING_RATE_FINE_TUNE,
        'input_size': INPUT_SIZE,
        'train_limit': TRAIN_LIMIT,
        'model_save_path': MODEL_SAVE_PATH,
        'device': str(device),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }

    meta_lines = [f"{k}: {v}" for k, v in metadata.items()]
    meta_text = "\n".join(meta_lines)

    # Leave room on the right for metadata and place it there
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.text(0.8, 0.5, meta_text, fontsize=8, ha='left', va='center', family='monospace')

    img_path = os.path.join(run_dir, f"training_history.png")
    fig.savefig(img_path)
    print(f"Saved training+accuracy plot to {img_path}")

    # Also save a compact accuracy-only figure
    acc_path = os.path.join(run_dir, f"accuracy_history.png")
    fig2, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(history['train_acc'], label='train_acc')
    ax.plot(history['val_acc'], label='val_acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(acc_path)
    plt.close(fig2)

    # Save CSV with metadata and per-epoch metrics
    csv_path = os.path.join(run_dir, f"evaluation.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # metadata section
        writer.writerow(['metadata_key', 'metadata_value'])
        for k, v in metadata.items():
            writer.writerow([k, v])
        writer.writerow([])
        # per-epoch metrics
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        for i in range(len(history['loss'])):
            writer.writerow([
                i + 1,
                history['loss'][i],
                history['val_loss'][i],
                history['train_acc'][i],
                history['val_acc'][i]
            ])
    print(f"Saved evaluation CSV to {csv_path}")

    plt.show()
except Exception as e:
    print('Could not create evaluation outputs:', e)