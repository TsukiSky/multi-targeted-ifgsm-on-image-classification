import os


class Configuration:
    PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
    CACHE_PATH = os.path.join(PROJECT_PATH, 'cache')
    IMAGE_PATH = os.path.join(PROJECT_PATH, 'dataset', 'images')
    DATA_ENTRY_PATH = os.path.join(PROJECT_PATH, 'dataset', 'data_entry_2017.csv')
    VICTIM_MODEL_PATH = os.path.join(PROJECT_PATH, 'experiment', 'victim_model')
    PERTURBED_IMAGE_PATH = os.path.join(PROJECT_PATH, 'experiment', 'evaluation', 'perturbed_images')

    CNN_MODEL_PATH = os.path.join(VICTIM_MODEL_PATH, 'cnn', 'chest_xray_cnn.pth')
    RESNET_MODEL_PATH = os.path.join(VICTIM_MODEL_PATH, 'resnet', 'chest_xray_resnet.pth')
    VIT_MODEL_PATH = os.path.join(VICTIM_MODEL_PATH, 'vit', 'vit.pth')
