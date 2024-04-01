"""
This scripts contains the evaluation for mdels
"""
import os

import torch
import torch.nn as nn
from torchvision import transforms, models

from config import Configuration
from dataset.dataset import ChestXrayDataset


MODEL_PATH = os.path.join(Configuration.VICTIM_MODEL_PATH, "resnet", "chest_xray_resnet18.pth")

# Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 is the input size for ResNet
    transforms.ToTensor(),  # convert images to PyTorch tensors
])

dataset = ChestXrayDataset(transform=transform)

# Load the model
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, dataset.get_num_classes())

state_dict = torch.load(MODEL_PATH)
model.load_state_dict(state_dict)

model.eval()
correct = 0
total = 0
threshold = 0.5

with torch.no_grad():
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.unsqueeze(0)
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > threshold).int()
        if torch.equal(predicted, label):
            correct += 1
        total += 1

print(f"Accuracy of the model on the {total} test images: {100 * correct / total}%")
