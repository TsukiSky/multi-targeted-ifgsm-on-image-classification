import numpy as np

from dataset import ChestXrayDataset
from torchvision import transforms, models
import torch
import torch.nn as nn

from config import Configuration
from experiment.victim_model.vit.vit import ViT
from experiment.victim_model.cnn_two_layer.cnn_two_layer import TwoLayerCNN
from experiment.victim_model.cnn_three_layer.cnn_three_layer import ThreeLayerCNN
from experiment.evaluation.evaluator import Evaluator
from experiment.evaluation.generator import Generator, AttackMethod

import warnings


warnings.filterwarnings("ignore")

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

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
cnn_model.load_state_dict(torch.load(Configuration.CNN_MODEL_PATH, map_location=device))
cnn_model.eval()
cnn_generator = Generator(cnn_model, AttackMethod.BOTH)
print("Loaded CNN model:", Configuration.CNN_MODEL_PATH)

cnn_three_layer_model = ThreeLayerCNN(image_input_channels=3, num_classes=dataset.get_num_classes())
cnn_three_layer_model.load_state_dict(torch.load(Configuration.CNN_THREE_LAYER_MODEL_PATH, map_location=device))
cnn_three_layer_model.eval()
cnn_three_layer_generator = Generator(cnn_three_layer_model, AttackMethod.BOTH)
print("Loaded CNN model:", Configuration.CNN_THREE_LAYER_MODEL_PATH)

resnet_model = models.resnet18(pretrained=False)
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, dataset.get_num_classes())
resnet_model.load_state_dict(torch.load(Configuration.RESNET_MODEL_PATH, map_location=device))
resnet_model.eval()
resnet_generator = Generator(resnet_model, AttackMethod.BOTH)
print("Loaded RESNET model:", Configuration.RESNET_MODEL_PATH)

vit_model = ViT(in_channels=3, patch_size=16, embedding_size=768, img_size=224, num_heads=4, num_layers=4,
                num_classes=15)
vit_model.load_state_dict(torch.load(Configuration.VIT_MODEL_PATH, map_location=device))
vit_model.eval()
vit_generator = Generator(vit_model, AttackMethod.BOTH)
print("Loaded VIT model:", Configuration.VIT_MODEL_PATH)

evaluator = Evaluator(None)  # No need for generating methods

cnn_results = []
cnn_three_layer_results = []
resnet_results = []
vit_results = []

for i in range(TEST_SAMPLES):
    image, label = test_dataset[i]
    print("#### Sample:", i, " Evaluating ... ####")
    cnn_itfgsm_image, cnn_mt_ifgsm_image = cnn_generator.generate(image, label, ITER, EPSILON, PERCENTAGE)
    cnn_three_layer_itfgsm_image, cnn_three_layer_mt_ifgsm_image = cnn_three_layer_generator.generate(image, label, ITER, EPSILON, PERCENTAGE)
    resnet_itfgsm_image, resnet_mt_ifgsm_image = resnet_generator.generate(image, label, ITER, EPSILON, PERCENTAGE)
    vit_itfgsm_image, vit_mt_ifgsm_image = vit_generator.generate(image, label, ITER, EPSILON, PERCENTAGE)

    cnn_metrics = evaluator.evaluate_stealthiness(image, cnn_itfgsm_image, cnn_mt_ifgsm_image)
    cnn_three_layer_metrics = evaluator.evaluate_stealthiness(image, cnn_three_layer_itfgsm_image, cnn_three_layer_mt_ifgsm_image)
    resnet_metrics = evaluator.evaluate_stealthiness(image, resnet_itfgsm_image, resnet_mt_ifgsm_image)
    vit_metrics = evaluator.evaluate_stealthiness(image, vit_itfgsm_image, vit_mt_ifgsm_image)

    cnn_results.append(cnn_metrics)
    cnn_three_layer_results.append(cnn_three_layer_metrics)
    resnet_results.append(resnet_metrics)
    vit_results.append(vit_metrics)
print("###################################")
print("Saving Results")
cnn_results = np.array(cnn_results)
cnn_three_layer_results = np.array(cnn_three_layer_results)
resnet_results = np.array(resnet_results)
vit_results = np.array(vit_results)

np.savetxt("cnn_stealthy_results.csv", cnn_results, delimiter=",",
           header="ITFGSM Hash Distance, "
                  "MT-IFGSM Hash Distance, "
                  "L2 Distance ITFGSM, "
                  "L2 Distance MT-IFGSM,"
                  "SSIM ITFGSM,"
                  "SSIM MT-IFGSM",
           comments='')
np.savetxt("cnn_three_layer_stealthy_results.csv", cnn_three_layer_results, delimiter=",",
           header="ITFGSM Hash Distance, "
                  "MT-IFGSM Hash Distance, "
                  "L2 Distance ITFGSM, "
                  "L2 Distance MT-IFGSM,"
                  "SSIM ITFGSM,"
                  "SSIM MT-IFGSM",
           comments='')
np.savetxt("resnet_stealthy_results.csv", resnet_results, delimiter=",",
           header="ITFGSM Hash Distance, "
                  "MT-IFGSM Hash Distance, "
                  "L2 Distance ITFGSM, "
                  "L2 Distance MT-IFGSM, "
                  "SSIM ITFGSM, "
                  "SSIM MT-IFGSM",
           comments='')
np.savetxt("vit_stealthy_results.csv", vit_results, delimiter=",",
           header="ITFGSM Hash Distance, "
                  "MT-IFGSM Hash Distance, "
                  "L2 Distance ITFGSM, "
                  "L2 Distance MT-IFGSM, "
                  "SSIM ITFGSM, "
                  "SSIM MT-IFGSM",
           comments='')

print("###################################")
cnn_mean = np.mean(cnn_results, axis=0)
cnn_three_layer_mean = np.mean(cnn_results, axis=0)
resnet_mean = np.mean(resnet_results, axis=0)
vit_mean = np.mean(vit_results, axis=0)
print("CNN Average Metrics:")
print("ITFGSM Hash Distance:", cnn_mean[0])
print("MT-IFGSM Hash Distance:", cnn_mean[1])
print("L2 Distance ITFGSM:", cnn_mean[2])
print("L2 Distance MT-IFGSM:", cnn_mean[3])
print("SSIM ITFGSM:", cnn_mean[4])
print("SSIM MT-IFGSM:", cnn_mean[5])
print("###################################")

print("CNN Three Layer Average Metrics:")
print("ITFGSM Hash Distance:", cnn_three_layer_mean[0])
print("MT-IFGSM Hash Distance:", cnn_three_layer_mean[1])
print("L2 Distance ITFGSM:", cnn_three_layer_mean[2])
print("L2 Distance MT-IFGSM:", cnn_three_layer_mean[3])
print("SSIM ITFGSM:", cnn_three_layer_mean[4])
print("SSIM MT-IFGSM:", cnn_three_layer_mean[5])
print("###################################")

print("ResNet Average Metrics:")
print("ITFGSM Hash Distance:", resnet_mean[0])
print("MT-IFGSM Hash Distance:", resnet_mean[1])
print("L2 Distance ITFGSM:", resnet_mean[2])
print("L2 Distance MT-IFGSM:", resnet_mean[3])
print("SSIM ITFGSM:", resnet_mean[4])
print("SSIM MT-IFGSM:", resnet_mean[5])
print("###################################")

print("ViT Average Metrics:")
print("ITFGSM Hash Distance:", vit_mean[0])
print("MT-IFGSM Hash Distance:", vit_mean[1])
print("L2 Distance ITFGSM:", vit_mean[2])
print("L2 Distance MT-IFGSM:", vit_mean[3])
print("SSIM ITFGSM:", vit_mean[4])
print("SSIM MT-IFGSM:", vit_mean[5])
print("###################################")
print("Finished")
