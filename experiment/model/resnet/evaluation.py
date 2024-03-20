"""
This scripts contains the evaluation for mdels
"""
import torch
import torch.nn as nn
from torchvision import transforms, models

from dataset.dataset import ChestXrayDataset


MODEL_PATH = "./resnet/chest_xray_resnet18.pth"

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
with torch.no_grad():
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.unsqueeze(0)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == label).sum().item()

print(f"Accuracy of the model on the {total} test images: {100 * correct / total}%")
