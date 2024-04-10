import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from dataset.dataset import ChestXrayDataset
from config import Configuration
from sklearn.metrics import precision_score, f1_score

MODEL_PATH = os.path.join(Configuration.VICTIM_MODEL_PATH, "resnet", "chest_xray_resnet.pth")

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

predictions = []
labels = []
threshold = 0.5

with torch.no_grad():
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.unsqueeze(0)
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > threshold).int()
        predictions.append(predicted)
        labels.append(label)

predictions = torch.cat(predictions)
labels = torch.stack(labels)

accuracy = (predictions == labels).float().mean().item()
precision = precision_score(labels.cpu(), predictions.cpu(), average='macro')
f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
