from PIL import Image
import imagehash
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch


class Evaluator:
    def __init__(self, model, log_values=False):
        self.model = model
        self.log_values = log_values

    def tensor_to_pil(self, tensor_image):
        tensor_image = tensor_image.cpu()
        if tensor_image.dim() == 4 and tensor_image.shape[0] == 1:
            tensor_image = tensor_image.squeeze(0)
        if tensor_image.dim() == 3 and tensor_image.shape[0] == 1:
            tensor_image = tensor_image.squeeze(0)

        tensor_image = tensor_image.mul(255).byte()
        if tensor_image.dim() == 3:
            tensor_image = tensor_image.permute(1, 2, 0)

        if tensor_image.dim() == 2:
            return Image.fromarray(tensor_image.numpy(), 'L')
        else:
            return Image.fromarray(tensor_image.numpy())

    def evaluate_stealthiness(self, original_image, itfgsm_image, mt_itfgsm_image):
        hash_original = imagehash.phash(self.tensor_to_pil(original_image))
        hash_itfgsm = imagehash.phash(self.tensor_to_pil(itfgsm_image))
        hash_mt_itfgsm = imagehash.phash(self.tensor_to_pil(mt_itfgsm_image))

        original_itfgsm_distance = hash_original - hash_itfgsm
        original_mt_itfgsm_distance = hash_original - hash_mt_itfgsm

        original_image = np.array(original_image)
        itfgsm_image = np.array(itfgsm_image.detach().numpy())
        mt_itfgsm_image = np.array(mt_itfgsm_image.detach().numpy())

        l2_distance_itfgsm = np.linalg.norm(original_image - itfgsm_image).round(2)
        l2_distance_mt_itfgsm = np.linalg.norm(original_image - mt_itfgsm_image).round(2)

        itfgsm_image = itfgsm_image.squeeze(0)
        mt_itfgsm_image = mt_itfgsm_image.squeeze(0)
        original_image_last = original_image.transpose(1, 2, 0)
        itfgsm_image_last = itfgsm_image.transpose(1, 2, 0)
        mt_itfgsm_image_last = mt_itfgsm_image.transpose(1, 2, 0)

        ssim_itfgsm = ssim(original_image_last, itfgsm_image_last, channel_axis=-1, data_range=1.0)
        ssim_mt_itfgsm = ssim(original_image_last, mt_itfgsm_image_last, channel_axis=-1, data_range=1.0)

        if self.log_values:
            print("###################### Hash Sensitive Difference ######################")
            print("Original and ITFGSM Image Hash Sensitive Difference:", original_itfgsm_distance)
            print("Original and MT-ITFGSM Image Hash Sensitive Difference:", original_mt_itfgsm_distance)
            print("############################# L2 Distance #############################")
            print("L2 Distance between Original and ITFGSM Image:", l2_distance_itfgsm)
            print("L2 Distance between Original and MT-ITFGSM Image:", l2_distance_mt_itfgsm)
            print("############################# SSIM #############################")
            print("SSIM between Original and ITFGSM Image:", ssim_itfgsm)
            print("SSIM between Original and MT-ITFGSM Image:", ssim_mt_itfgsm)

        return original_itfgsm_distance, original_mt_itfgsm_distance, l2_distance_itfgsm, l2_distance_mt_itfgsm, ssim_itfgsm, ssim_mt_itfgsm

    def evaluate_attack_performance(self, adversarial_image, original_label, multi_classification_threshold=0.5):
        adversarial_output = self.model(adversarial_image)

        adversarial_output = (torch.sigmoid(adversarial_output) > multi_classification_threshold).float()
        accuracy_per_sample = (adversarial_output == original_label).all(dim=1).float()
        hamming_loss_per_sample = (adversarial_output != original_label).float().mean(dim=1)
        return accuracy_per_sample, hamming_loss_per_sample
