import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class CobotDataset(Dataset):
    def __init__(self, root_dir, label, transform=None, num_segments=5, mode='random'):
        """
        Args:
            root_dir (str): Directory containing sequence folders.
            label (int): Label for the dataset (e.g., 0 for start, 1 for stop).
            transform (callable): Transformations to apply to each frame.
            num_segments (int): Number of frames to sample per sequence.
            mode (str): Frame selection mode ('5_second', '2_second', 'random').
        """
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.sequence_folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir)]
        self.num_segments = num_segments
        self.mode = mode

    def __len__(self):
        return len(self.sequence_folders)

    def __getitem__(self, idx):
        folder_path = self.sequence_folders[idx]
        frame_paths = sorted([os.path.join(folder_path, frame) for frame in os.listdir(folder_path)])

        if self.mode == '5_second':
            # Evenly sample frames across the full 5-second sequence
            frame_indices = torch.linspace(0, len(frame_paths) - 1, self.num_segments).long()
        elif self.mode == '2_second':
            # Select the first num_segments frames from the first 2 seconds (at 3 fps)
            frame_indices = torch.arange(0, min(self.num_segments, len(frame_paths))).long()
        elif self.mode == 'random':
            # 50% random, 25% 5_second, 25% 2_second
            prob = torch.rand(1).item()
            if prob < 0.33:
                # Random sampling
                frame_indices = torch.randperm(len(frame_paths))[:self.num_segments]
            elif prob < 0.66:
                # 5_second sampling
                frame_indices = torch.linspace(0, len(frame_paths) - 1, self.num_segments).long()
            else:
                # 2_second sampling
                frame_indices = torch.arange(0, min(self.num_segments, len(frame_paths))).long()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose from '5_second', '2_second', 'random'.")


        frames = [self.transform(Image.open(frame_paths[i]).convert('RGB')) for i in frame_indices]
        frames = torch.stack(frames)  # Shape: (num_segments, 3, height, width)
        return frames, self.label
