import os
import torch
from torchvision import transforms
from dataset.dataset import ChestXrayDataset
from cnn import TwoLayerCNN
from config import Configuration
from sklearn.metrics import precision_score, f1_score

MODEL_PATH = os.path.join(Configuration.VICTIM_MODEL_PATH, "cnn", "chest_xray_cnn.pth")

torch.manual_seed(100)
# Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 is the input size
    transforms.ToTensor(),  # convert images to PyTorch tensors
])

dataset = ChestXrayDataset(transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Load the model
model = TwoLayerCNN(image_input_channels=3, num_classes=dataset.get_num_classes())
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

predictions = []
labels = []

threshold = 0.5

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
hamming_loss = (predictions != labels).float().mean().item()

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Hamming Loss: {hamming_loss:.4f}")
