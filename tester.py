import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from ops.models import TSN
from cobot_dataset.dataset_manager import CobotDataHandler
from sklearn.metrics import confusion_matrix

class TSMTester:
    def __init__(self, start_dir, stop_dir, model_path, num_classes, num_segments, batch_size, device, seed=42):
        self.start_dir = start_dir
        self.stop_dir = stop_dir
        self.model_path = model_path
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.batch_size = batch_size
        self.device = device

        self._set_seed(seed)

        # Data Handler
        self.data_handler = CobotDataHandler(start_dir, stop_dir, seed)

        # Initialize model/load weights
        self.model = TSN(num_classes, num_segments, modality='RGB', base_model='resnet50', consensus_type='avg')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def test(self, split_mode="random", sequence_interval="5_second"):
        # Get test loader based on split mode
        _, _, test_loader = self.data_handler.get_dataloaders(
            batch_size=self.batch_size, 
            split_mode=split_mode,
            sequence_interval = sequence_interval
        )

        test_correct, test_total = 0, 0
        total_inference_time = 0
        all_test_labels, all_test_preds = [], []
        num_batches = len(test_loader)
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                # Flatten sequences
                batch_size, num_segments, _, _, _ = sequences.size()
                sequences = sequences.view(batch_size * num_segments, 3, 224, 224)

                # Model inference
                start_time = time.time()
                outputs = self.model(sequences)
                end_time = time.time()
                total_inference_time += (end_time - start_time)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

                # Collect predictions and labels
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(predicted.cpu().numpy())

        # Calculate metrics
        test_accuracy = test_correct / test_total
        test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
        test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')
        test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
        conf_matrix = confusion_matrix(all_test_labels, all_test_preds)
        average_inference_time = (total_inference_time / num_batches) * 1000

        # Print results
        print(f"Test Results - Split Mode: {split_mode}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1 Score: {test_f1:.4f}")

        print("\nConfusion Matrix:")
        print(f"{'':<10}Predicted Start  Predicted Stop")
        print(f"Actual Start {conf_matrix[0, 0]:>10} {conf_matrix[0, 1]:>15}")
        print(f"Actual Stop  {conf_matrix[1, 0]:>10} {conf_matrix[1, 1]:>15}")

        print(f"Average Inference Time: {average_inference_time:.2f} ms")
        

# Main execution
if __name__ == "__main__":
    start_dir = '/data/scratch/ec23984/cobot_data/all_start_sequences'
    stop_dir = '/data/scratch/ec23984/cobot_data/all_stop_sequences'

    # Paths for pre-trained models
    sequential_model_path = 'experiments/sequential_split_final/cobot_tsm_model.pth'
    random_split_model_path = 'experiments/random_split_final/cobot_tsm_model.pth'

    sequential_tester = TSMTester(
        start_dir=start_dir,
        stop_dir=stop_dir,
        model_path=sequential_model_path,
        num_classes=2,
        num_segments=5,
        batch_size=6,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42
    )

    random_split_tester = TSMTester(
        start_dir=start_dir,
        stop_dir=stop_dir,
        model_path=random_split_model_path,
        num_classes=2,
        num_segments=5,
        batch_size=6,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=42
    )

    log_file = open("experiments/results/test_results.log", "w")
    sys.stdout = log_file

    print("\n*** Results for Random-Split Model ***")
    random_split_tester.test(split_mode="random", sequence_interval="5_second")

    print("\n=== Results for Sequential-Split Model ===")
    sequential_tester.test(split_mode="sequential", sequence_interval="5_second")

    log_file.close()
