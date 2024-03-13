from attack.attack import Attack

import torch
import torch.nn.functional as F


class Itfgsm(Attack):
    """
    ITFGSM: Iterative Targeted Fast Gradient Sign Method
    """

    def __init__(self, model, epsilon=0.01, iters=40, target=False):
        super(Itfgsm, self).__init__("ITFGSM", model)
        self.epsilon = epsilon
        self.iters = iters
        self.target = target

    def target_attack(self, image, ori_label, target_label):
        image = image.clone().detach().requires_grad_(True)
        for i in range(self.iters):
            output = self.model(image)
            loss = F.cross_entropy(output, target_label)
            self.model.zero_grad()
            loss.backward()
            image_grad = image.grad.data
            image = image + self.epsilon * image_grad.sign()
            image = torch.clamp(image, 0, 1)
            image = image.clone().detach().requires_grad_(True)
        return image

    def untargeted_attack(self, image, ori_label):
        image = image.clone().detach().requires_grad_(True)
        for i in range(self.iters):
            output = self.model(image)
            loss = F.cross_entropy(output, ori_label)
            self.model.zero_grad()
            loss.backward()
            image_grad = image.grad.data
            image = image - self.epsilon * image_grad.sign()
            image = torch.clamp(image, 0, 1)
            image = image.clone().detach().requires_grad_(True)
        return image
