import os
import torch
from torchvision import transforms
from dataset.dataset import ChestXrayDataset
from cnn import TwoLayerCNN
from config import Configuration
from sklearn.metrics import precision_score, f1_score

MODEL_PATH = os.path.join(Configuration.VICTIM_MODEL_PATH, "cnn", "chest_xray_cnn.pth")

# Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 is the input size
    transforms.ToTensor(),  # convert images to PyTorch tensors
])

dataset = ChestXrayDataset(transform=transform)

# Load the model
model = TwoLayerCNN(image_input_channels=3, num_classes=dataset.get_num_classes())
model.load_state_dict(torch.load(MODEL_PATH))
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