from enum import Enum
from attack.itfgsm import Itfgsm
from attack.mtifgsm import MtIfgsm
from torchvision import transforms

import os


class AttackMethod(Enum):
    ITFGSM = "itfgsm"
    MT_IFGSM = "mt_ifgsm"
    BOTH = "both"


class Generator:
    def __init__(self, model, attack_method: AttackMethod):
        self.model = model
        model.eval()
        self.attack_method = attack_method
        self.itfgsm = Itfgsm(model)
        self.mt_ifgsm = MtIfgsm(model)

    def generate(self, image, original_label, iter=5, epsilon=0.001, percentage=0.2, save_image=False, path=""):
        itfgsm_image = None
        mt_ifgsm_image = None
        if self.attack_method == AttackMethod.ITFGSM:
            itfgsm_image = self.itfgsm.untargeted_attack(image, original_label, epsilon, iter)
        elif self.attack_method == AttackMethod.MT_IFGSM:
            mt_ifgsm_image = self.mt_ifgsm.mt_ifgsm_attack(image, original_label, epsilon, iter, percentage)
        else:
            itfgsm_image = self.itfgsm.untargeted_attack(image, original_label, epsilon, iter)
            mt_ifgsm_image = self.mt_ifgsm.mt_ifgsm_attack(image, original_label, epsilon, iter, percentage)

        if save_image:
            transforms.ToPILImage()(image.squeeze(0)).save(os.path.join(path, "original.png"))
            if itfgsm_image is not None:
                transforms.ToPILImage()(itfgsm_image.squeeze(0)).save(os.path.join(path, "itfgsm.png"))
            if mt_ifgsm_image is not None:
                transforms.ToPILImage()(mt_ifgsm_image.squeeze(0)).save(os.path.join(path, "mt_ifgsm.png"))

        return itfgsm_image, mt_ifgsm_image
