import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ops.models import TSN
from PIL import Image

# Configuration
start_state_dataset = '/data/scratch/ec23984/cobot_data/start_sequences'
stop_state_dataset = '/data/scratch/ec23984/cobot_data/stop_sequences'
batch_size = 8
num_classes = 2  # "start" and "stop"
num_segments = 5  # Number of frames per sequence
num_epochs = 20  # Keep it small for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CobotDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.sequence_folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir)]
    
    def __len__(self):
        return len(self.sequence_folders)
    
    def __getitem__(self, idx):
        folder_path = self.sequence_folders[idx]
        frame_paths = sorted([os.path.join(folder_path, frame) for frame in os.listdir(folder_path)])

        # Load frames as images and apply transformations
        frames = [self.transform(Image.open(frame)) for frame in frame_paths[:num_segments]]
        frames = torch.stack(frames)  # Shape: (num_segments, 3, height, width)
        return frames, self.label


# Define data preprocessing
transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
start_dataset = CobotDataset(start_state_dataset, label=0, transform=transform)
stop_dataset = CobotDataset(stop_state_dataset, label=1, transform=transform)

# Combine datasets
dataset = torch.utils.data.ConcatDataset([start_dataset, stop_dataset])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize TSM Model
model = TSN(num_classes, num_segments, modality='RGB', base_model='resnet50', consensus_type='avg')
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
        for sequences, labels in t:
            sequences, labels = sequences.to(device), labels.to(device)  # Move to GPU

            # Flatten sequences for TSM input
            batch_size, num_segments, _, _, _ = sequences.size()
            sequences = sequences.view(batch_size * num_segments, 3, 224, 224)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            t.set_postfix(loss=running_loss / (t.n + 1), accuracy=correct / total)

print("Finished Training!") 