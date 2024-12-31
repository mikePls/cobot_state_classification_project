import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from ops.models import TSN
from cobot_dataset.dataset_manager import CobotDataHandler

class TSMTrainer:
    def __init__(self, train_loader, val_loader, test_loader, num_classes, num_segments, batch_size, num_epochs, device, experiment_dir, learning_rate=1e-4, step_size=8, gamma=0.2, dropout=0.5):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.experiment_dir = experiment_dir
        self.learning_rate = learning_rate

        # Model initialisation..
        self.model = TSN(num_classes, num_segments, modality='RGB', base_model='resnet50', consensus_type='avg', dropout=dropout)
        self.model.to(device)

        # Loss, Optimiser, Scheduler
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        # TensorBoard
        os.makedirs(experiment_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=experiment_dir)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as t:
                for sequences, labels in t:
                    sequences, labels = sequences.to(self.device), labels.to(self.device)

                    # Flatten for TSM input
                    batch_size, num_segments, _, _, _ = sequences.size()
                    sequences = sequences.view(batch_size * num_segments, 3, 224, 224)

                    # Forward pass
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update metrics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    t.set_postfix(loss=running_loss / (t.n + 1), accuracy=correct / total)

                self.scheduler.step()

            # Logs to TensorBoard
            train_accuracy = correct / total
            self.writer.add_scalar('Training Loss', running_loss / len(self.train_loader), epoch)
            self.writer.add_scalar('Training Accuracy', train_accuracy, epoch)

            # Validate
            self.validate(epoch)

        # Save
        model_path = os.path.join(self.experiment_dir, 'cobot_tsm_model.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                # Flatten sequences
                batch_size, num_segments, _, _, _ = sequences.size()
                sequences = sequences.view(batch_size * num_segments, 3, 224, 224)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Collect metrics
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # Validation Metrics
        val_loss /= len(self.val_loader)
        val_accuracy = val_correct / val_total
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Logs to TensorBoard
        self.writer.add_scalar('Validation Loss', val_loss, epoch)
        self.writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
        self.writer.add_scalar('Validation Precision', precision, epoch)
        self.writer.add_scalar('Validation Recall', recall, epoch)
        self.writer.add_scalar('Validation F1', f1, epoch)

        print(f"Epoch {epoch+1}/{self.num_epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    def test(self):
        self.model.eval()
        test_correct, test_total = 0, 0
        all_test_labels, all_test_preds = [], []
        with torch.no_grad():
            for sequences, labels in self.test_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                # Flatten sequences
                batch_size, num_segments, _, _, _ = sequences.size()
                sequences = sequences.view(batch_size * num_segments, 3, 224, 224)

                outputs = self.model(sequences)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

                # Metrics
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(predicted.cpu().numpy())

        # Test Metrics
        test_accuracy = test_correct / test_total
        test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
        test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')
        test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')

        print(f"Test Results - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")


if __name__ == "__main__":
    start_dir = '/data/scratch/ec23984/cobot_data/all_start_sequences'
    stop_dir = '/data/scratch/ec23984/cobot_data/all_stop_sequences'


    data_handler = CobotDataHandler(start_dir, stop_dir,seed=42)

    train_loader, val_loader, test_loader = data_handler.get_dataloaders(batch_size=6, split_mode="random")

    trainer = TSMTrainer(
        train_loader, val_loader, test_loader,
        num_classes=2, 
        num_segments=5,
        batch_size=6,
        num_epochs=40, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        experiment_dir='./experiments'
    )

    trainer.train()
    trainer.test()