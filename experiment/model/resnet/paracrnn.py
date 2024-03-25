import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

from dataset.dataset import ChestXrayDataset

class ParallelCRNN(nn.Module):
    def __init__(self, image_input_channels, text_input_size, num_classes):
        super(ParallelCRNN, self).__init__()
        
        # Image branch
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Text branch
        self.text_embedding = nn.Embedding(text_input_size, 128)
        self.text_rnn = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(128, num_classes)
        
    def forward(self, image_input, text_input):
        # Image branch
        image_features = self.image_conv(image_input)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Text branch
        text_embedded = self.text_embedding(text_input)
        text_output, _ = self.text_rnn(text_embedded)
        text_features = torch.mean(text_output, dim=1)  
        
        # Fusion
        fused_features = torch.cat((image_features, text_features), dim=1)
        output = self.fusion_layer(fused_features)
        
        return output

if __name__ == "__main__":
    # Configuration
    image_input_channels = 3
    text_input_size = 1000
    num_classes = 15
    model_path = "parallel_crnn_model.pth"
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
    model = ParallelCRNN(image_input_channels, text_input_size, num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
