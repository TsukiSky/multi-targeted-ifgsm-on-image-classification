from attack import Attack

import torch
import torch.nn as nn


class Itfgsm(Attack):
    """
    ITFGSM: Iterative Targeted Fast Gradient Sign Method
    """
    def __init__(self, model):
        super(Itfgsm, self).__init__("ITFGSM", model)
        self.model.eval()

    def target_attack(self, image, target_label, epsilon=0.01, iters=10):
        # check if the image is batched
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        for i in range(iters):
            image = image.clone().detach().requires_grad_(True)
            output = self.model(image)[0]
            loss = nn.BCEWithLogitsLoss()(output, target_label)
            loss.backward()
            image_grad = image.grad.data
            image = image - epsilon * image_grad.sign()
            image = torch.clamp(image, 0, 1)
        return image

    def untargeted_attack(self, image, ori_label, epsilon=0.01, iters=10):
        # check if the image is batched
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        for i in range(iters):
            image = image.clone().detach().requires_grad_(True)
            output = self.model(image)[0]
            loss = nn.BCEWithLogitsLoss()(output, ori_label)
            loss.backward()
            image_grad = image.grad.data
            image = image + epsilon * image_grad.sign()
            image = torch.clamp(image, 0, 1)
        return image
