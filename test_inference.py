import torch
import time
from torchvision.transforms import transforms
from PIL import Image
from ops.models import TSN
import os

# Config
model_path = './experiments/cobot_tsm_model.pth'  
num_classes = 2  # "start" and "stop"
num_segments = 5  # Number of frames per sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
model = TSN(num_classes, num_segments, modality='RGB', base_model='resnet50', consensus_type='avg')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully.")

sequence_folder = '/data/scratch/ec23984/cobot_data/start_sequences/sequence_001'
frame_paths = sorted([f"{sequence_folder}/{frame}" for frame in sorted(os.listdir(sequence_folder))[:num_segments]])

frames = []
for frame_path in frame_paths:
    img = Image.open(frame_path).convert('RGB')
    img = transform(img)
    frames.append(img)
frames = torch.stack(frames)  # Shape: (num_segments, 3, height, width)

# Add batch dim/move to device
input_tensor = frames.unsqueeze(0).to(device)  # Shape: (1, num_segments, 3, height, width)

# Flatten for TSM
batch_size, num_segments, _, _, _ = input_tensor.size()
input_tensor = input_tensor.view(batch_size * num_segments, 3, 224, 224)

# Inf. time
with torch.no_grad():
    start_time = time.time()
    output = model(input_tensor)
    end_time = time.time()

    _, predicted = torch.max(output, 1)
    print(f"Predicted Class: {predicted.item()}")  # (0 for stop, 1 for start)

inference_time = (end_time - start_time) * 1000 
print(f"Inference Time: {inference_time:.2f} ms")
