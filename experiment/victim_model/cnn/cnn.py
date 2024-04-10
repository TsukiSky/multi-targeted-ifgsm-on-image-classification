import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset import ChestXrayDataset


class TwoLayerCNN(nn.Module):
    def __init__(self, image_input_channels, num_classes):
        super(TwoLayerCNN, self).__init__()

        # Image branch
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fusion layer
        self.fusion_layer = nn.Linear(128 * 56 * 56, num_classes)

    def forward(self, image_input):
        # Image branch
        image_features = self.image_conv(image_input)
        image_features = image_features.view(image_features.size(0), -1)

        # Fusion
        output = self.fusion_layer(image_features)

        return output


if __name__ == "__main__":
    # Configuration
    image_input_channels = 3
    num_classes = 15
    model_path = "chest_xray_cnn.pth"
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ChestXrayDataset(transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    model = TwoLayerCNN(image_input_channels, num_classes)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
