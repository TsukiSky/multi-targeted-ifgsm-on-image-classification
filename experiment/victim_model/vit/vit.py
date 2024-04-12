import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset import ChestXrayDataset


class PositionalEmbedding(nn.Module):
    def __init__(self, dimension=768, max_length=1000):
        super().__init__()
        positional_embedding = torch.zeros(max_length, dimension)

        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))

        positional_embedding[:, 0::2] = torch.sin(positions * div_term)
        positional_embedding[:, 1::2] = torch.cos(positions * div_term)

        positional_embedding = positional_embedding.unsqueeze(0)

        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self, x):
        x = x + self.positional_embedding[:, :x.size(1)]
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.positions = PositionalEmbedding(embedding_size, num_patches + 1)

    def forward(self, x):
        x = self.projection(x)
        b = x.shape[0]  # b is the batch num
        cls_tokens = self.cls_token.expand(b, -1, -1)
        # print(x.shape)
        # print(cls_tokens.shape)
        x = x.permute(0, 2, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.positions(x)
        return x


class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_size=768, img_size=224, num_heads=4, num_layers=4,
                 num_classes=15):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_size, img_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.permute(1, 0, 2)  # batch, sequence, features -> sequence, batch, features
        x = self.transformer_encoder(x)
        x = x[0, :, :]
        x = self.classifier(x)
        return x


INPUT_CHANNELS = 3
NUM_CLASSES = 15
MODEL_PATH = "vit.pth"
BATCH_SIZE = 32
PATCH_SIZE = 16
EMBEDDING_SIZE = 768

TRANSFORMER_HEADS_NUM = 4

TRANSFORMER_LAYERS_NUM = 4

if __name__ == "__main__":
    torch.manual_seed(100)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 224x224 is the input size for ResNet
        transforms.ToTensor(),  # convert images to PyTorch tensors
    ])

    dataset = ChestXrayDataset(transform=transform)

    # split the dataset into training and validation
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 2. Prepare the model and optimizer
    model = ViT(in_channels=INPUT_CHANNELS, patch_size=PATCH_SIZE, embedding_size=EMBEDDING_SIZE, img_size=224,
                num_heads=TRANSFORMER_HEADS_NUM, num_layers=TRANSFORMER_LAYERS_NUM, num_classes=NUM_CLASSES)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    torch.save(model.state_dict(), MODEL_PATH)
