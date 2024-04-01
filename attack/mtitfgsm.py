import torch

from attack.attack import Attack
from attack.itfgsm import Itfgsm


class MtItfgsm(Attack):
    """
    MTITFGSM: Multi-Targeted Iterative Fast Gradient Sign Method
    """

    def __init__(self, model):
        super(MtItfgsm, self).__init__("MTITFGSM", model)
        self.model.eval()

    def target_attack(self, image, target_label, epsilon=0.01, iters=10):
        itfgsm = Itfgsm(self.model)
        return itfgsm.target_attack(image, target_label, epsilon, iters)

    def untargeted_attack(self, image, ori_label, epsilon=0.01, iters=10):
        itfgsm = Itfgsm(self.model)
        return itfgsm.untargeted_attack(image, ori_label, epsilon, iters)

    def stealthy_untargeted_attack(self, image, ori_label, epsilon=0.01, iters=10, percentage=0.5):
        """
        stealthy untargeted attack is designed to make the multi-targeted attack more stealthy
        - instead of perturbing the image to the completely wrong direction, we identify the top-k
          most likely classes and perturb the image towards those directions
        - the percentage parameter controls the percentage of the top-k classes to consider,
          e.g., if percentage=0.1, we consider the top 10% of the classes to perturb the image
        """
        image = image.clone().detach().requires_grad_(True)
        image = image.unsqueeze(0)
        output = self.model(image)
        probabilities = torch.sigmoid(output)

        # identify the top-k most likely wrongly predicted classes
        inverse_label = 1 - ori_label  # inverse the original label

        # get the loss between the inverse label and the predicted probability
        probability_loss = torch.abs(probabilities - inverse_label)

        # get the top-k classes that are most likely to be wrongly predicted
        top_k_classes = torch.topk(probability_loss, int(percentage * 15), largest=False).indices
        # create the target label
        target_label = ori_label.clone()
        # inverse the top-k classes
        target_label[top_k_classes] = 1 - target_label[top_k_classes]

        return self.target_attack(image, target_label, epsilon, iters)
