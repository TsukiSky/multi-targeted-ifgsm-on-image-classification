# Resnet 18

## Model Architecture
A complex CNN-architecture model with residual connections [2, 3]. 
We directly load the architecture of this model provided by the [torchvision.models API](https://pytorch.org/hub/pytorch_vision_resnet/).

![Resnet 18 Architecture](/images/resnet.png)

## Performance
The model has been evaluated using standard classification metrics:

- **Accuracy:** 41.76%
- **Precision:** 0.2185
- **F1 Score:** 0.1582
- **Hamming Loss:** 0.0721

These metrics reflect the preliminary results obtained under our current experimental setup.