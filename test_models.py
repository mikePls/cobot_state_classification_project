import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
from ops.models import TSN
from cobot_dataset.dataset_manager import CobotDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configuration
test_start_sequences = '/data/scratch/ec23984/cobot_data/test_start_sequences'
test_stop_sequences = '/data/scratch/ec23984/cobot_data/test_stop_sequences'
num_classes = 2  # "start" and "stop"
num_segments = 5  # Number of frames per sequence
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/data/home/ec23984/code/cobot_project/temporal_shift_module/experiments/consecutive_split_model/cobot_tsm_model.pth'

# Define Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Datasets
test_start_dataset = CobotDataset(test_start_sequences, label=0, transform=transform, mode='5_second')
test_stop_dataset = CobotDataset(test_stop_sequences, label=1, transform=transform, mode='5_second')

# Combine Datasets
test_dataset = torch.utils.data.ConcatDataset([test_start_dataset, test_stop_dataset])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load Model
model = TSN(num_classes, num_segments, modality='RGB', base_model='resnet50', consensus_type='avg')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully.")

# Testing Loop
all_test_labels, all_test_preds = [], []
mislabeled_sequences = []
test_correct, test_total = 0, 0
total_inference_time = 0
num_batches = len(test_loader)

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        # Flatten sequences
        batch_size, num_segments, _, _, _ = sequences.size()
        sequences = sequences.view(batch_size * num_segments, 3, 224, 224)

        # Measure inference time
        start_time = time.time()
        outputs = model(sequences)
        end_time = time.time()

        total_inference_time += (end_time - start_time)

        # Predictions and metrics
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        # Collect for metrics
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(predicted.cpu().numpy())

        # Track mislabeled sequences
        for idx in range(len(labels)):
            if labels[idx] != predicted[idx]:
                sequence_name = test_dataset.datasets[0].sequence_folders[idx].split("/")[-1]  # Extract folder name
                mislabeled_sequences.append((sequence_name, labels[idx].item(), predicted[idx].item()))
                if len(mislabeled_sequences) >= 5:  # Stop after 5
                    break

# Calculate Metrics
test_accuracy = test_correct / test_total
test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')
test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
average_inference_time = (total_inference_time / num_batches) * 1000  # To ms
conf_matrix = confusion_matrix(all_test_labels, all_test_preds)

print(f"Test Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")


# Print Confusion Matrix
print("\nConfusion Matrix:")
print(f"{'':<10}Predicted Start  Predicted Stop")
print(f"Actual Start {conf_matrix[0, 0]:>10} {conf_matrix[0, 1]:>15}")
print(f"Actual Stop  {conf_matrix[1, 0]:>10} {conf_matrix[1, 1]:>15}")

# Visualize Confusion Matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Start", "Stop"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.show()

print(f"Average Inference Time: {average_inference_time:.2f} ms")

