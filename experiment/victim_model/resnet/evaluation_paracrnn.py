import os
import torch
import torch.nn as nn
from torchvision import transforms
from dataset.dataset import ChestXrayDataset
from model.paracrnn import ParallelCRNN
from config import Configuration

MODEL_PATH = os.path.join(Configuration.VICTIM_MODEL_PATH, "paracrnn", "chest_xray_paracrnn_model.pth")

# Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 is the input size
    transforms.ToTensor(),  # convert images to PyTorch tensors
])

dataset = ChestXrayDataset(transform=transform)

# Load the model
model = ParallelCRNN(image_input_channels=3, num_classes=dataset.get_num_classes())
model.load_state_dict(torch.load(MODEL_PATH))
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
