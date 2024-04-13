import numpy as np

from dataset import ChestXrayDataset
from torchvision import transforms, models
import torch
import torch.nn as nn

from config import Configuration
from experiment.victim_model.vit.vit import ViT
from experiment.victim_model.cnn.cnn import TwoLayerCNN
from experiment.evaluation.evaluator import Evaluator
from experiment.evaluation.generator import Generator, AttackMethod

import warnings

warnings.filterwarnings("ignore")

TEST_SAMPLES = 1000
ITER = 10
EPSILON = 0.001
PERCENTAGE = 0.2

print("Set test samples to 1000")
torch.manual_seed(100)
print("Set random seed to 100")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 is the input size
    transforms.ToTensor(),  # convert images to PyTorch tensors
])

dataset = ChestXrayDataset(transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print("Loaded dataset: ChestXrayDataset")

cnn_model = TwoLayerCNN(image_input_channels=3, num_classes=dataset.get_num_classes())
cnn_model.load_state_dict(torch.load(Configuration.CNN_MODEL_PATH))
cnn_model.eval()
cnn_generator = Generator(cnn_model, AttackMethod.BOTH)
print("Loaded CNN model:", Configuration.CNN_MODEL_PATH)

resnet_model = models.resnet18(pretrained=False)
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, dataset.get_num_classes())
resnet_model.load_state_dict(torch.load(Configuration.RESNET_MODEL_PATH))
resnet_model.eval()
resnet_generator = Generator(resnet_model, AttackMethod.BOTH)
print("Loaded RESNET model:", Configuration.RESNET_MODEL_PATH)

vit_model = ViT(in_channels=3, patch_size=16, embedding_size=768, img_size=224, num_heads=4, num_layers=4,
                num_classes=15)
vit_model.load_state_dict(torch.load(Configuration.VIT_MODEL_PATH))
vit_model.eval()
vit_generator = Generator(vit_model, AttackMethod.BOTH)
print("Loaded VIT model:", Configuration.VIT_MODEL_PATH)

cnn_evaluator = Evaluator(cnn_model)  # No need for generating methods
resnet_evaluator = Evaluator(resnet_model)
vit_evaluator = Evaluator(vit_model)

cnn_results = np.zeros((TEST_SAMPLES, 4))
resnet_results = np.zeros((TEST_SAMPLES, 4))
vit_results = np.zeros((TEST_SAMPLES, 4))

for i in range(TEST_SAMPLES):
    image, label = test_dataset[i]
    print("#### Sample:", i, " Evaluating ... ####")
    cnn_itfgsm_image, cnn_mt_itfgsm_image = cnn_generator.generate(image, label, ITER, EPSILON, PERCENTAGE)
    cnn_itfgsm_per_sample_accuracy, cnn_itfgsm_per_sample_hamming_loss = cnn_evaluator.evaluate_attack_performance(
        cnn_itfgsm_image, label)
    cnn_mt_itfgsm_per_sample_accuracy, cnn_mt_itfgsm_per_sample_hamming_loss = cnn_evaluator.evaluate_attack_performance(
        cnn_mt_itfgsm_image, label)
    cnn_results[i] = [cnn_itfgsm_per_sample_accuracy.item(), cnn_itfgsm_per_sample_hamming_loss.item(),
                      cnn_mt_itfgsm_per_sample_accuracy.item(), cnn_mt_itfgsm_per_sample_hamming_loss.item()]

    resnet_itfgsm_image, resnet_mt_itfgsm_image = resnet_generator.generate(image, label, ITER, EPSILON, PERCENTAGE)
    resnet_itfgsm_per_sample_accuracy, resnet_itfgsm_per_sample_hamming_loss = resnet_evaluator.evaluate_attack_performance(
        resnet_itfgsm_image, label)
    resnet_mt_itfgsm_per_sample_accuracy, resnet_mt_itfgsm_per_sample_hamming_loss = resnet_evaluator.evaluate_attack_performance(
        resnet_mt_itfgsm_image, label)
    resnet_results[i] = [resnet_itfgsm_per_sample_accuracy.item(), resnet_itfgsm_per_sample_hamming_loss.item(),
                            resnet_mt_itfgsm_per_sample_accuracy.item(), resnet_mt_itfgsm_per_sample_hamming_loss.item()]

    vit_itfgsm_image, vit_mt_itfgsm_image = vit_generator.generate(image, label, ITER, EPSILON, PERCENTAGE)
    vit_itfgsm_per_sample_accuracy, vit_itfgsm_per_sample_hamming_loss = vit_evaluator.evaluate_attack_performance(
        vit_itfgsm_image, label)
    vit_mt_itfgsm_per_sample_accuracy, vit_mt_itfgsm_per_sample_hamming_loss = vit_evaluator.evaluate_attack_performance(
        vit_mt_itfgsm_image, label)
    vit_results[i] = [vit_itfgsm_per_sample_accuracy.item(), vit_itfgsm_per_sample_hamming_loss.item(),
                        vit_mt_itfgsm_per_sample_accuracy.item(), vit_mt_itfgsm_per_sample_hamming_loss.item()]

print("###################################")
cnn_itfgsm_accuracy = cnn_results[:, 0].mean()
cnn_itfgsm_hamming_loss = cnn_results[:, 1].mean()
cnn_mt_itfgsm_accuracy = cnn_results[:, 2].mean()
cnn_mt_itfgsm_hamming_loss = cnn_results[:, 3].mean()

resnet_itfgsm_accuracy = resnet_results[:, 0].mean()
resnet_itfgsm_hamming_loss = resnet_results[:, 1].mean()
resnet_mt_itfgsm_accuracy = resnet_results[:, 2].mean()
resnet_mt_itfgsm_hamming_loss = resnet_results[:, 3].mean()

vit_itfgsm_accuracy = vit_results[:, 0].mean()
vit_itfgsm_hamming_loss = vit_results[:, 1].mean()
vit_mt_itfgsm_accuracy = vit_results[:, 2].mean()
vit_mt_itfgsm_hamming_loss = vit_results[:, 3].mean()
print("Summary:")
print("CNN ITFGSM Accuracy:", cnn_itfgsm_accuracy)
print("CNN ITFGSM Hamming Loss:", cnn_itfgsm_hamming_loss)
print("CNN MT-ITFGSM Accuracy:", cnn_mt_itfgsm_accuracy)
print("CNN MT-ITFGSM Hamming Loss:", cnn_mt_itfgsm_hamming_loss)
print("-----------------------------------")
print("RESNET ITFGSM Accuracy:", resnet_itfgsm_accuracy)
print("RESNET ITFGSM Hamming Loss:", resnet_itfgsm_hamming_loss)
print("RESNET MT-ITFGSM Accuracy:", resnet_mt_itfgsm_accuracy)
print("RESNET MT-ITFGSM Hamming Loss:", resnet_mt_itfgsm_hamming_loss)
print("-----------------------------------")
print("VIT ITFGSM Accuracy:", vit_itfgsm_accuracy)
print("VIT ITFGSM Hamming Loss:", vit_itfgsm_hamming_loss)
print("VIT MT-ITFGSM Accuracy:", vit_mt_itfgsm_accuracy)
print("VIT MT-ITFGSM Hamming Loss:", vit_mt_itfgsm_hamming_loss)
print("###################################")
print("Saving Results")
data = np.array([[
    cnn_itfgsm_accuracy,
    cnn_itfgsm_hamming_loss,
    cnn_mt_itfgsm_accuracy,
    cnn_mt_itfgsm_hamming_loss,
    resnet_itfgsm_accuracy,
    resnet_itfgsm_hamming_loss,
    resnet_mt_itfgsm_accuracy,
    resnet_mt_itfgsm_hamming_loss,
    vit_itfgsm_accuracy,
    vit_itfgsm_hamming_loss,
    vit_mt_itfgsm_accuracy,
    vit_mt_itfgsm_hamming_loss
]])

np.savetxt("performance_results.csv",
           data,
           delimiter=",",
           fmt='%.4f',
           header="CNN ITFGSM Accuracy, "
                  "CNN ITFGSM Hamming Loss, "
                  "CNN MT-ITFGSM Accuracy, "
                  "CNN MT-ITFGSM Hamming Loss, "
                  "RESNET ITFGSM Accuracy, "
                  "RESNET ITFGSM Hamming Loss, "
                  "RESNET MT-ITFGSM Accuracy, "
                  "RESNET MT-ITFGSM Hamming Loss, "
                  "ViT ITFGSM Accuracy, "
                  "ViT ITFGSM Hamming Loss, "
                  "ViT MT-ITFGSM Accuracy, "
                  "ViT MT-ITFGSM Hamming Loss",
           comments='')
print("###################################")
print("Finished!")
