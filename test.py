import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. Define Constants and Configuration ---

TEST_DIR = "./dataset/test" # The final, unseen test set
MODEL_PATH = "./arthritis_classifier.pth"
BATCH_SIZE = 32
INPUT_SIZE = 224

# --- 2. Set up the 'mps' Device ---

if not torch.backends.mps.is_available():
    print("MPS not available. Using CPU.")
    device = torch.device("cpu")
else:
    print("MPS is available. Using M1 GPU.")
    device = torch.device("mps")

# --- 3. Custom Dataset Class ---
# We must include the *exact same* Dataset class used in training
# to ensure data is loaded correctly.

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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# --- 4. Define Data Transform and DataLoader ---

# Use the 'val' transform from the training script
test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the Dataset and DataLoader
test_dataset = ArthritisDataset(root_dir=TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Data loaded: {len(test_dataset)} test images from 'auto_test'.")

# --- 5. Re-define and Load the Model ---

# We must define the same model architecture to load the weights
model = models.resnet18(weights=None) # We don't need pre-trained weights, just the architecture
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# Load the saved state dictionary
model.load_state_dict(torch.load(MODEL_PATH))

# Move the model to the 'mps' device
model = model.to(device)

# --- 6. Evaluation Loop ---

print("Starting final evaluation on the 'auto_test' set...")
model.eval()  # Set model to evaluation mode
all_labels = []
all_preds = []

with torch.no_grad():  # Disable gradient calculation
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        
        # Forward pass
        outputs = model(inputs.to(device))
        
        # Convert logits to binary predictions (0 or 1)
        preds = torch.round(torch.sigmoid(outputs))
        
        # Store labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# --- 7. Calculate and Print Metrics ---

# Calculate metrics using sklearn
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("\n--- Final Test Results ('auto_test') ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Correct positive predictions / All positive predictions)")
print(f"Recall:    {recall:.4f} (Correct positive predictions / All actual positives)")
print(f"F1-Score:  {f1:.4f} (Harmonic mean of Precision and Recall)")