from abc import abstractmethod


class Attack:
    def __init__(self, name: str, model):
        self.name = name
        self.model = model
        self.device = next(model.parameters()).device

    @abstractmethod
    def target_attack(self, image, ori_label, target_label):
        pass

    @abstractmethod
    def untargeted_attack(self, image, ori_label):
        pass
