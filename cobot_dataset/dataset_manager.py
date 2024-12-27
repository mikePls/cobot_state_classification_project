import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torchvision import transforms
from PIL import Image

class CobotDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label, transform=None, sequence_interval="random", num_segments=5):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.num_segments = num_segments
        self.sequence_folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir)]
        self.sequence_interval = sequence_interval

    def __len__(self):
        return len(self.sequence_folders)

    def __getitem__(self, idx):
        folder_path = self.sequence_folders[idx]
        frame_paths = sorted([os.path.join(folder_path, frame) for frame in os.listdir(folder_path)])

        if self.sequence_interval == "random":
            selected_frames = random.sample(frame_paths, self.num_segments)
        elif self.sequence_interval == "5_second":
            selected_frames = frame_paths[:self.num_segments]
        elif self.sequence_interval == "2_second":
            stride = max(len(frame_paths) // self.num_segments, 1)
            selected_frames = frame_paths[::stride][:self.num_segments]
        else:
            raise ValueError("Invalid mode specified: Choose 'random', '5_second', or '2_second'.")

        # Load/transform frames
        frames = [self.transform(Image.open(frame).convert('RGB')) for frame in selected_frames]
        frames = torch.stack(frames)  # Shape: (num_segments, 3, height, width)
        return frames, self.label

class CobotDataHandler:
    def __init__(self, start_dir, stop_dir, seed=42):
        self.start_dir = start_dir
        self.stop_dir = stop_dir
        self.seed = seed
        self.split_indices = {}  # Dictionary to remember splits
        self._set_seed()

        self.train_transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
            ], p=0.8),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.test_transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def get_dataloaders(self, batch_size=8, split_mode="random",
                        train_val_split=0.8, num_test_sequences=120, sequence_interval="5_second"):
        start_dataset = CobotDataset(self.start_dir, label=0, transform=self.train_transform, sequence_interval=sequence_interval)
        stop_dataset = CobotDataset(self.stop_dir, label=1, transform=self.train_transform, sequence_interval=sequence_interval)

        full_dataset = ConcatDataset([start_dataset, stop_dataset])

        if split_mode == "random":
            # Random split into train, val, and test
            train_size = int(train_val_split * len(full_dataset))
            val_size = int((1 - train_val_split) * len(full_dataset) / 2)
            test_size = len(full_dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

            self.split_indices["train"] = train_dataset.indices
            self.split_indices["val"] = val_dataset.indices
            self.split_indices["test"] = test_dataset.indices

        # elif split_mode == "sequential":
        #     # Sequential split: first num_test_sequences for test, rest for train and val
        #     test_start_dataset = Subset(start_dataset, range(num_test_sequences))
        #     test_stop_dataset = Subset(stop_dataset, range(num_test_sequences))

        #     train_start_dataset = Subset(start_dataset, range(num_test_sequences, len(start_dataset)))
        #     train_stop_dataset = Subset(stop_dataset, range(num_test_sequences, len(stop_dataset)))

        #     train_val_dataset = ConcatDataset([train_start_dataset, train_stop_dataset])
        #     test_dataset = ConcatDataset([test_start_dataset, test_stop_dataset])

        #     train_size = int(train_val_split * len(train_val_dataset))
        #     val_size = len(train_val_dataset) - train_size

        #     train_dataset, val_dataset = random_split(
        #         train_val_dataset,
        #         [train_size, val_size],
        #         generator=torch.Generator().manual_seed(self.seed)
        #     )

        #     self.split_indices["train"] = [i for i in range(len(train_dataset))]
        #     self.split_indices["val"] = [i for i in range(len(val_dataset))]
        #     self.split_indices["test"] = [i for i in range(len(test_dataset))]
        elif split_mode == "sequential":
            # Sequential split: first sequences for train, last num_test_sequences for test
            train_start_dataset = Subset(start_dataset, range(len(start_dataset) - num_test_sequences))
            train_stop_dataset = Subset(stop_dataset, range(len(stop_dataset) - num_test_sequences))

            test_start_dataset = Subset(start_dataset, range(len(start_dataset) - num_test_sequences, len(start_dataset)))
            test_stop_dataset = Subset(stop_dataset, range(len(stop_dataset) - num_test_sequences, len(stop_dataset)))

            train_val_dataset = ConcatDataset([train_start_dataset, train_stop_dataset])
            test_dataset = ConcatDataset([test_start_dataset, test_stop_dataset])

            train_size = int(train_val_split * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size

            train_dataset, val_dataset = random_split(
                train_val_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

            self.split_indices["train"] = [i for i in range(len(train_dataset))]
            self.split_indices["val"] = [i for i in range(len(val_dataset))]
            self.split_indices["test"] = [i for i in range(len(test_dataset))]

        else:
            raise ValueError("Invalid split_mode: Choose 'random' or 'sequential'.")

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(split_mode == "random"), num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    def save_split_indices(self, path):
        torch.save(self.split_indices, path)

    def load_split_indices(self, path):
        self.split_indices = torch.load(path)
