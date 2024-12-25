import os
import time
import random
import cv2
import torch
import numpy as np
from torchvision import transforms
from moviepy.editor import ImageSequenceClip
from ops.models import TSN 

class CobotDemoGenerator:
    def __init__(self, start_dir, stop_dir, model_path, num_class, num_segments, device):
        self.start_dir = start_dir
        self.stop_dir = stop_dir
        self.model_path = model_path
        self.num_class = num_class
        self.num_segments = num_segments
        self.device = device

        # Load model
        self.model = TSN(num_class, num_segments, modality='RGB', base_model='resnet50', consensus_type='avg')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_sequence(self, sequence_path):
        frames = sorted([os.path.join(sequence_path, f) for f in os.listdir(sequence_path)])
        processed_frames = []
        for frame in frames:
            image = cv2.imread(frame)
            if image is None:
                raise ValueError(f"Image {frame} could not be loaded. Check the path or file.")
            image_rgb = image[:, :, ::-1].copy()  # Convert BGR to RGB and ensure positive strides
            processed_frames.append(self.transform(image_rgb))
        return torch.stack(processed_frames).to(self.device)

    def predict_sequence(self, sequence_tensor):
        batch_size, num_segments, _, _, _ = sequence_tensor.size()
        sequence_tensor = sequence_tensor.view(batch_size * num_segments, 3, 224, 224)  # Flatten for TSM
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def select_random_sequences(self, num_samples):
        # Select random sequences
        start_sequences = random.sample(os.listdir(self.start_dir), num_samples // 2)
        stop_sequences = random.sample(os.listdir(self.stop_dir), num_samples // 2)

        start_paths = [os.path.join(self.start_dir, seq) for seq in start_sequences]
        stop_paths = [os.path.join(self.stop_dir, seq) for seq in stop_sequences]

        # Combine paths and labels into a list of tuples
        combined = list(zip(start_paths, [0] * (num_samples // 2))) + list(zip(stop_paths, [1] * (num_samples // 2)))

        random.shuffle(combined)
        sequences, labels = zip(*combined)

        return list(sequences), list(labels)


    def generate_video(self, sequence_dirs, labels, output_path):
        video_frames = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1

        for sequence_path, label in zip(sequence_dirs, labels):
            frames = sorted([os.path.join(sequence_path, f) for f in os.listdir(sequence_path)])
            sequence_tensor = self.load_sequence(sequence_path).unsqueeze(0)  # Add batch dimension
            prediction = self.predict_sequence(sequence_tensor)[0]

            # Display the first 13 frames for 3 seconds (90 frames at 30 FPS)
            for frame_path in frames[:13]:
                image = cv2.imread(frame_path)  # Read the frame for visualization
                if image is None:
                    raise ValueError(f"Image {frame_path} could not be loaded.")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for video output

                # Resize frame to double size (e.g., 448x448 for 224x224 input)
                image_rgb = cv2.resize(image_rgb, (448, 448), interpolation=cv2.INTER_LINEAR)

                for _ in range(7):  # Duplicate each frame 7 times to make them last longer
                    video_frames.append(image_rgb)

            # Display the last frame with prediction and ground truth for 2 seconds (60 frames at 30 FPS)
            final_frame = cv2.imread(frames[-1])  # Read the last frame
            final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)

            # Resize the last frame to double size
            final_frame_rgb = cv2.resize(final_frame_rgb, (448, 448), interpolation=cv2.INTER_LINEAR)

            # Annotate the last frame
            prediction_text = f"Predicted: {'working' if prediction == 0 else 'stopped'}"
            label_text = f"Ground Truth: {'working' if label == 0 else 'stopped'}"
            color = (0, 255, 0) if prediction == label else (255, 0, 0)  # Green for correct, red for wrong
            cv2.putText(final_frame_rgb, prediction_text, (10, 30), font, font_scale, color, font_thickness, cv2.LINE_AA)
            cv2.putText(final_frame_rgb, label_text, (10, 60), font, font_scale, color, font_thickness, cv2.LINE_AA)

            for _ in range(60):  # Hold for 2 seconds
                video_frames.append(final_frame_rgb)

        clip = ImageSequenceClip(video_frames, fps=40)
        clip.write_videofile(output_path, codec="libx264", audio=False)


if __name__ == "__main__":
    start_dir = '/data/scratch/ec23984/cobot_data/all_start_sequences'
    stop_dir = '/data/scratch/ec23984/cobot_data/all_stop_sequences'
    model_path = 'experiments/random_split_final/cobot_tsm_model.pth'
    output_video = 'cobot_demo_video.mp4'

    # Num of sequences to visualise
    num_samples = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualizer = CobotDemoGenerator(start_dir, stop_dir, model_path, num_class=2, num_segments=5, device=device)

    # Select random sequences
    sequences, labels = visualizer.select_random_sequences(num_samples)

    visualizer.generate_video(sequences, labels, output_video)
