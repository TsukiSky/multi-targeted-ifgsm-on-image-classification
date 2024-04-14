# MT-IFGSM Attack on Image Classification Deep Neural Networks

A Course Project for SUTD 50.039 Theory and Practice of Deep Learning (2024 Spring)

Check out our [report](doc/project_report.pdf).

## Team Members

[Xiang Siqi](https://github.com/TsukiSky) 1004875

[Kishen](https://github.com/K15H3N) 1005885

[Luah Shi Hui](https://github.com/ShiHui21) 1005512

[Liu Yu](https://github.com/Dr123Ake) 1005621

## Introduction

Traditional adversarial attack methodologies on image classification tasks have primarily focused on single-target prediction tasks, where the aim is to deceive the model into misclassifying an image as an incorrect label. While effective at exploiting vulnerabilities in deep learning models, this traditional approach does not fully capture the complexity of real-world applications, where decisions are neither binary nor singular. In contrast, multi-target classification tasks, prevalent in sectors such as medical imaging and multi-class object detection, require the model to discern among multiple correct categories, adding more complexity to the classification challenge.

To this end, we introduce the Multi-Targeted Iterative Fast Gradient Sign Method (MT-IFGSM), an innovative adversarial attack methodology designed specifically for multi-targeted image classification tasks.

## Setup Environment

```
# clone this repository
git clone https://github.com/TsukiSky/multi-targeted-ifgsm-on-image-classification.git

# Set up Python virtual environment
python3 -m venv venv && source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

## Setup Dataset

We use a portion of the [NIH Chest X-ray dataset](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset) as our dataset. The dataset contains 25000 images of chest X-rays, with 14 different diseases. You can download the dataset from [here](https://drive.google.com/file/d/1GTWlKciNyAwJCjz8Id9kCnnpuKev4Sxb/view?usp=drive_link).

After downloading the dataset, put the images folder under the dataset package. The directory of the dataset package should be  structured as follows:
├─dataset  
│  ├─images  
│  ├─script

We provide an overview of our dataset [here](https://github.com/TsukiSky/multi-targeted-ifgsm-on-image-classification/blob/main/dataset/script/dataset_analyze.ipynb).

## Victim Models

We provide four victim models.

* [2-layer CNN](/experiment/victim_model/cnn_two_layer): A straightforward Convolutional Neural Network with two convolutional layers followed by a fully connected layer.
* [3-layer CNN](experiment/victim_model/cnn_three_layer): A three-layer CNN model with a fully connected layer.
* [ResNet18](experiment/victim_model/resnet): A CNN-architecture model with residual connections.
* [Simple ViT](experiment/victim_model/vit): Our implementation of a simplified Vision Transformer model.

You can find them under [victim models](experiment/victim_model). We have trained them using our training dataset.

## MT-IFGSM Attack

You can find the attack's implementation at [MT-IFGSM](attack/mtifgsm.py). We also provide an implementation of traditional ITFGSM attack at [ITFGSM](attack/itfgsm.py).

## Evaluation

We provide a generator API and an evaluator API for you to produce adversarial samples and evaluate the attacks' performance.

To generate an adversarial sample, run:

``````python
# Generator
from experiment.evaluation.generator import Generator, AttackMethod

model = # load the victim model
image = # original multi-channel image
original_label = # original label of the sample

generator = Generator(model, AttackMethod.MT_IFGSM) # to generate MT_IFGSM samples

_, mt_ifgsm_image = cnn_generator.generate(image, original_label)
``````

To evaluate the stealthiness and performance of the attack, run:

``````python
# import Evaluator
from experiment.evaluation.evaluator import Evaluator

model = # load the victim model
evaluator = Evaluator(model)

original_image, itfgsm_image, mt_ifgsm_image = # multi-channel images
original_label = # original label of the sample

# 1. evaluate the stealthiness of the samples
metrics = evaluator.evaluate_stealthiness(original_image, itfgsm_image, mt_ifgsm_image)

# 2. evaluate the peformance of the attack
accuracy, hamming_loss = evaluator.evaluate_attack_performance(mt_ifgsm_image, original_label)
``````

You can check out our [evaluation](experiment/evaluation/script/evaluate) and [generation](experiment/evaluation/script/generate) scripts.

## License

Our project is licensed under the [MIT License](LICENSE.txt).
