"""
This scripts contains the evaluation for mdels
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from config import Configuration
from experiment.victim_model.cnn.cnn_three_layer import ThreeLayerCNN
from dataset.dataset import ChestXrayDataset
from sklearn.metrics import precision_score, f1_score

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

MODEL_PATH = os.path.join(Configuration.VICTIM_MODEL_PATH, "cnn", "chest_xray_cnn_three_layer.pth")

torch.manual_seed(100)
# Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 is the input size for ResNet
    transforms.ToTensor(),  # convert images to PyTorch tensors
])

dataset = ChestXrayDataset(transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Load the model
model = ThreeLayerCNN(image_input_channels=3, num_classes=dataset.get_num_classes())
model.to(device)

state_dict = torch.load(MODEL_PATH)
model.load_state_dict(state_dict=state_dict)

model.eval()

threshold = 0.5

predictions = []
labels = []

with torch.no_grad():
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        image = image.unsqueeze(0)
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > threshold).int()
        predictions.append(predicted)
        labels.append(label)


predictions = torch.cat(predictions)
labels = torch.stack(labels)

accuracy_per_sample = (predictions == labels).all(dim=1).float()
accuracy = accuracy_per_sample.mean().item()
precision = precision_score(labels.cpu(), predictions.cpu(), average='macro')
f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

hamming_loss = (predictions != labels).float().mean().item()
print(f"Hamming Loss: {hamming_loss:.4f}")