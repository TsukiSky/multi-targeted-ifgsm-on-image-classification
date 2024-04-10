import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

from dataset.dataset import ChestXrayDataset


cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# Step 1. Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 is the input size for ResNet
    transforms.ToTensor(),  # convert images to PyTorch tensors
])

dataset = ChestXrayDataset(transform=transform)

# split the dataset into training and validation
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2. Prepare the model and optimizer
model = models.resnet18(pretrained=True)

num_features = model.fc.in_features
# add a fully connected layer with number of classes as output
model.fc = nn.Linear(num_features, dataset.get_num_classes())
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 3. Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()

# Step 4. Save the model
model_path = "chest_xray_resnet.pth"

torch.save(model.state_dict(), model_path)
