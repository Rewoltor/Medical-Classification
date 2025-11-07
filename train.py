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

# Configuration from previous paper
LEARNING_RATE_NEW_LAYER = 0.01    
LEARNING_RATE_FINE_TUNE = 0.001   
BATCH_SIZE = 32
NUM_EPOCHS = 6 # maybe 20-30 epocs on few images
MODEL_SAVE_PATH = "./arthritis_classifier.pth" 

# 1: Updated Data Paths**
TRAIN_DIR = "./dataset/train"
VAL_DIR = "./dataset/val"         # Using 'val' for validation
INPUT_SIZE = 224

# --- 2. Set up the 'mps' Device ---

if not torch.backends.mps.is_available():
    print("MPS not available. Using CPU.")
    device = torch.device("cpu")
else:
    print("MPS is available. Using M1 GPU.")
    device = torch.device("mps")

# --- 3. Custom Dataset Class (Modified to limit training data) ---

class ArthritisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        label_map = {'0': 0, '2': 1, '3': 1, '4': 1}
        
        for class_folder in os.listdir(root_dir):
            if class_folder in label_map:
                label = label_map[class_folder]
                class_path = os.path.join(root_dir, class_folder)
                
                for img_path in glob.glob(os.path.join(class_path, "*.png")):
                    self.file_paths.append(img_path)
                    self.labels.append(label)
            
            elif class_folder == '1':
                print(f"Ignoring folder: {os.path.join(root_dir, class_folder)}")
            
            else:
                print(f"Warning: Found unexpected folder, ignoring: {class_folder}")

        # --- 2. MODIFICATION TO LIMIT TRAINING DATA ---
        
        # Combine paths and labels to shuffle them together
        temp_list = list(zip(self.file_paths, self.labels))
        random.shuffle(temp_list)
        
        # Unzip back into lists
        self.file_paths, self.labels = zip(*temp_list)
        
        # Convert from tuple back to list
        self.file_paths = list(self.file_paths)
        self.labels = list(self.labels)

        # Apply 600 image limit *only* to the training set
        limit = 400
        # Check if 'train' is in the directory path (to not limit 'val' set)
        if "train" in root_dir and len(self.file_paths) > limit:
            self.file_paths = self.file_paths[:limit]
            self.labels = self.labels[:limit]
            print(f"  -> Dataset at {root_dir} randomly limited to {limit} images.")
        # --- END OF MODIFICATION ---

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
        transforms.RandomRotation(10),     
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([ 
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = ArthritisDataset(root_dir=TRAIN_DIR, transform=data_transforms['train'])
val_dataset = ArthritisDataset(root_dir=VAL_DIR, transform=data_transforms['val']) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) 

print(f"Data loaded: {len(train_dataset)} train images, {len(val_dataset)} validation images.") # This will now show 600 train images

# --- 5. Define the ResNet-18 Model and Fine-Tuning Setup ---

model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last convolutional block (model.layer4)
for param in model.layer4.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

model = model.to(device)

# --- 6. Define Loss Function and Optimizer ---

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': LEARNING_RATE_NEW_LAYER},
    {'params': model.layer4.parameters(), 'lr': LEARNING_RATE_FINE_TUNE}
])

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

# --- Evaluation visualization ---
try:
    plt.figure(figsize=(6, 4), dpi=160)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    print('Saved training history plot to training_history.png')
    plt.show()
except Exception as e:
    print('Could not create training plot:', e)