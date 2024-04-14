import os

from PIL import Image
import imagehash
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import Configuration

ORIGINAL_IMAGE_PATH =os.path.join(Configuration.PERTURBED_IMAGE_PATH, "original_image.png")
ADVERSARIAL_IMAGE_PATH = os.path.join(Configuration.PERTURBED_IMAGE_PATH, "itfgsm.png")
STEALTHY_ADVERSARIAL_IMAGE_PATH = os.path.join(Configuration.PERTURBED_IMAGE_PATH, "mt_ifgsm.png")

if __name__ == "__main__":
    original_image = Image.open(ORIGINAL_IMAGE_PATH)
    adversarial_image = Image.open(ADVERSARIAL_IMAGE_PATH)
    stealthy_adversarial_image = Image.open(STEALTHY_ADVERSARIAL_IMAGE_PATH)

    # 1. Hash sensitive evaluation
    hash_original = imagehash.phash(original_image)
    hash_adversarial = imagehash.phash(adversarial_image)
    hash_stealthy_adversarial = imagehash.phash(stealthy_adversarial_image)

    original_adversarial_distance = hash_original - hash_adversarial
    original_stealthy_adversarial_distance = hash_original - hash_stealthy_adversarial

    print("###################### Hash Sensitive Difference ######################")
    print("Original and Adversarial Image Hash Sensitive Difference:", original_adversarial_distance)
    print("Original and Stealthy Adversarial Image Hash Sensitive Difference:", original_stealthy_adversarial_distance)

    # 2. Pixel-wise evaluation
    original_image = np.array(original_image)
    adversarial_image = np.array(adversarial_image)
    stealthy_adversarial_image = np.array(stealthy_adversarial_image)

    # l0_distance_adversarial = np.count_nonzero(original_image - adversarial_image)
    # l0_distance_stealthy_adversarial = np.count_nonzero(original_image - stealthy_adversarial_image)

    l2_distance_adversarial = np.linalg.norm(original_image - adversarial_image).round(2)
    l2_distance_stealthy_adversarial = np.linalg.norm(original_image - stealthy_adversarial_image).round(2)

    ssim_adversarial = ssim(original_image, adversarial_image, multichannel=True)
    ssim_stealthy_adversarial = ssim(original_image, stealthy_adversarial_image, multichannel=True)

    # print("############################# L0 Distance #############################")
    # print("L0 Distance between Original and Adversarial Image:", l0_distance_adversarial)
    # print("L0 Distance between Original and Stealthy Adversarial Image:", l0_distance_stealthy_adversarial)
    print("############################# L2 Distance #############################")
    print("L2 Distance between Original and Adversarial Image:", l2_distance_adversarial)
    print("L2 Distance between Original and Stealthy Adversarial Image:", l2_distance_stealthy_adversarial)
    print("############################# SSIM #############################")
    print("SSIM between Original and Adversarial Image:", ssim_adversarial)
    print("SSIM between Original and Stealthy Adversarial Image:", ssim_stealthy_adversarial)
