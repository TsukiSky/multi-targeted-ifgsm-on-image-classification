import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset import ChestXrayDataset

class ThreeLayerCNN(nn.Module):
    def __init__(self, image_input_channels, num_classes):
        super(ThreeLayerCNN, self).__init__()

        # Image branch
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fusion_layer = nn.Linear(256 * 28 * 28, num_classes)

    def forward(self, image_input):
        # Image branch
        image_features = self.image_conv(image_input)
        image_features = image_features.view(image_features.size(0), -1)

        # Fusion
        output = self.fusion_layer(image_features)

        return output

if __name__ == "__main__":

    torch.manual_seed(100)

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

    # Configurations
    image_input_channels = 3
    num_classes = dataset.get_num_classes()
    model_path = "chest_xray_cnn_three_layer.pth"
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 20

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 2. Prepare the model and optimizer
    model = ThreeLayerCNN(image_input_channels, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Step 3. Train the model
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        torch.save(model.state_dict(), f'chest_xray_cnn_three_layer_{epoch}_epochs.pth')

    # Step 4. Save the model
    torch.save(model.state_dict(), model_path)