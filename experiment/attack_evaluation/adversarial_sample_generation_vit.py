import os

from torchvision import transforms, models
import torch
import torch.nn as nn

from attack.mtitfgsm import MtItfgsm
from dataset.dataset import ChestXrayDataset

from config import Configuration
from experiment.victim_model.vit.vit import ViT

MODEL_PATH = os.path.join(Configuration.VICTIM_MODEL_PATH, "vit", "vit.pth")
SAVE_IMAGE = True
SAVE_IMAGE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images")
NUM_SAMPLES = 1
MULTI_CLASSIFICATION_THRESHOLD = 0.5
STEALTHY_ATTACK_PERCENTAGE = 0.4
EPSILON = 0.005
ITERS = 40

INPUT_CHANNELS = 3
NUM_CLASSES = 15
BATCH_SIZE = 32
PATCH_SIZE = 16
EMBEDDING_SIZE = 768

TRANSFORMER_HEADS_NUM = 4

TRANSFORMER_LAYERS_NUM = 4


if __name__ == "__main__":
    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 224x224 is the input size for ResNet
        transforms.ToTensor(),  # convert images to PyTorch tensors
    ])

    dataset = ChestXrayDataset(transform=transform)
    print("Loaded dataset: ChestXrayDataset")

    model = ViT(in_channels=INPUT_CHANNELS, patch_size=PATCH_SIZE, embedding_size=EMBEDDING_SIZE, img_size=224,
                num_heads=TRANSFORMER_HEADS_NUM, num_layers=TRANSFORMER_LAYERS_NUM, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Loaded model: " + MODEL_PATH)

    attack = MtItfgsm(model)

    for i in range(NUM_SAMPLES):
        image, label = dataset[i]
        image_untargeted = attack.untargeted_attack(image, label, epsilon=EPSILON, iters=ITERS)
        image_stealthy_untargeted = attack.stealthy_untargeted_attack(image, label, percentage=STEALTHY_ATTACK_PERCENTAGE, epsilon=EPSILON, iters=ITERS)

        if SAVE_IMAGE:
            transforms.ToPILImage()(image.squeeze(0)).save(os.path.join(SAVE_IMAGE_PATH, "original_image_vit_" + str(i) + ".png"))
            transforms.ToPILImage()(image_untargeted.squeeze(0)).save(os.path.join(SAVE_IMAGE_PATH, "untargeted_image_vit_" + str(i) + ".png"))
            transforms.ToPILImage()(image_stealthy_untargeted.squeeze(0)).save(os.path.join(SAVE_IMAGE_PATH, "stealthy_untargeted_image_vit_" + str(i) + ".png"))

        # evaluate the attack results
        model.eval()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        output = model(image)
        print(torch.sigmoid(output))
        output_untargeted = model(image_untargeted)
        print(torch.sigmoid(output_untargeted))
        output_stealthy_untargeted = model(image_stealthy_untargeted)
        print(torch.sigmoid(output_stealthy_untargeted))

        output = (torch.sigmoid(output) > MULTI_CLASSIFICATION_THRESHOLD).float()
        output_untargeted = (torch.sigmoid(output_untargeted) > MULTI_CLASSIFICATION_THRESHOLD).float()
        output_stealthy_untargeted = (torch.sigmoid(output_stealthy_untargeted) > MULTI_CLASSIFICATION_THRESHOLD).float()
        # output the index of the predicted class (index that not equal to 0)
        print("Original Image Prediction:", torch.nonzero(output, as_tuple=True)[1].tolist())
        print("Untargeted Attack Image Prediction:", torch.nonzero(output_untargeted, as_tuple=True)[1].tolist())
        print("Stealthy Untargeted Attack Image Prediction:", torch.nonzero(output_stealthy_untargeted, as_tuple=True)[1].tolist())
