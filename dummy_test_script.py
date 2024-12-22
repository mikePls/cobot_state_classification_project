from ops.models import TSN
import torch.nn.functional as F
import torch




num_classes = 2  # Your task: "start" and "stop"
modality = 'RGB'  # Input is video frames
arch = 'resnet50'  # Backbone model

# Initialize the TSM model
model = TSN(num_classes, 8, modality, base_model=arch, consensus_type='avg', img_feature_dim=256)
model.eval()

# Create dummy input data
batch_size = 1
num_segments = 8
height, width = 224, 224

# Random tensor mimicking RGB frames
dummy_input = torch.rand(batch_size, num_segments, 3, height, width)

# Perform a forward pass
with torch.no_grad():  # Disable gradient computation for inference
    output = model(dummy_input)
    print("Model output:", output)

probabilities = F.softmax(output, dim=1)
print("Class probabilities:", probabilities)
