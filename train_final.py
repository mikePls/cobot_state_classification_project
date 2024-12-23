import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.transforms import functional as F
from ops.models import TSN
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from cobot_dataset.dataset_manager_original import CobotDataset

# Configuration
start_dataset_dir = '/data/scratch/ec23984/cobot_data_sequential_split/train_start_sequences'
stop_dataset_dir = '/data/scratch/ec23984/cobot_data_sequential_split/train_stop_sequences'
test_start_dataset_dir = '/data/scratch/ec23984/cobot_data_sequential_split/test_start_sequences'
test_stop_dataset_dir = '/data/scratch/ec23984/cobot_data_sequential_split/test_stop_sequences'
batch_size = 6
num_classes = 2  # "start" and "stop"
num_segments = 5  # Number of frames per sequence
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
experiment_dir = './experiments'

# Ensure the experiments directory exists
os.makedirs(experiment_dir, exist_ok=True)

# set TensorBoard 
writer = SummaryWriter(log_dir=experiment_dir)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    ], p=0.8),  # Apply affine or perspective with 80% probability
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
,
])

test_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor and normalize to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

# Load Train Datasets
start_dataset = CobotDataset(start_dataset_dir, label=0, transform=transform, mode='random')
stop_dataset = CobotDataset(stop_dataset_dir, label=1, transform=transform, mode='random')

# Load Test Datasets
test_start_dataset = CobotDataset(test_start_dataset_dir, transform=test_transform,label=0, mode='5_second')
test_stop_dataset = CobotDataset(test_stop_dataset_dir, transform=test_transform, label=1, mode='5_second')

# Combine Datasets
full_dataset = torch.utils.data.ConcatDataset([start_dataset, stop_dataset])
full_test_dataset = torch.utils.data.ConcatDataset([test_start_dataset, test_stop_dataset])

# Train/Test/Validation Split
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset= random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize TSM Model
model = TSN(num_classes, num_segments,
            modality='RGB', base_model='resnet50',
            consensus_type='avg', dropout=0.5)
model.to(device)

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Learning Rate Scheduler
scheduler = StepLR(optimizer, step_size=8, gamma=0.2)  # Decay LR by 0.1 every 10 epochs

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
        for sequences, labels in t:
            sequences, labels = sequences.to(device), labels.to(device)

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
            t.set_postfix(loss=running_loss / (t.n + 1), accuracy=correct / total)

        scheduler.step()

    # Log training metrics to TensorBoard
    train_accuracy = correct / total
    writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)

    # Validation Loop
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Flatten sequences
            batch_size, num_segments, _, _, _ = sequences.size()
            sequences = sequences.view(batch_size * num_segments, 3, 224, 224)

            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            # Collect for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Compute Validation Metrics
    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Log validation metrics to TensorBoard
    writer.add_scalar('Validation Loss', val_loss / len(val_loader), epoch)
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
    writer.add_scalar('Validation Precision', precision, epoch)
    writer.add_scalar('Validation Recall', recall, epoch)
    writer.add_scalar('Validation F1', f1, epoch)

    print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Save Model
model_path = os.path.join(experiment_dir, 'cobot_tsm_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Close TensorBoard writer
writer.close()

# Test Results
model.eval()
test_correct, test_total = 0, 0
all_test_labels, all_test_preds = [], []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        # Flatten sequences
        batch_size, num_segments, _, _, _ = sequences.size()
        sequences = sequences.view(batch_size * num_segments, 3, 224, 224)

        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        # Collect for metrics
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(predicted.cpu().numpy())

test_accuracy = test_correct / test_total
test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')
test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')

print(f"Test Results - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
