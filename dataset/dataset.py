import torch
from torch.utils.data import Dataset
from config import Configuration
from PIL import Image
from torchvision import transforms
from collections import Counter

import pandas as pd

import os


class ChestXrayDataset(Dataset):
    """
    Dataset class for the Chest X-ray dataset
    The images are stored in the 'images' folder
    """

    def __init__(self, transform=None):
        self.images = [f for f in os.listdir(Configuration.IMAGE_PATH) if f.endswith(".png")]
        self.data_entry = pd.read_csv(Configuration.DATA_ENTRY_PATH)[:len(self.images)]
        self.transform = transform
        self.label_map = {
            'No Finding': 0,
            'Infiltration': 1,
            'Hernia': 2,
            'Pneumothorax': 3,
            'Mass': 4,
            'Cardiomegaly': 5,
            'Consolidation': 6,
            'Emphysema': 7,
            'Nodule': 8,
            'Atelectasis': 9,
            'Effusion': 10,
            'Fibrosis': 11,
            'Pneumonia': 12,
            'Edema': 13,
            'Pleural_Thickening': 14
        }

    def __len__(self):
        # return the number of png files in the images folder
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get the image and label for the given index
        :param idx: of the image
        :return: in tensor and label
        """
        image_name = self.images[idx]
        image_path = os.path.join(Configuration.IMAGE_PATH, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        # get the label for the image
        labels = self.data_entry.loc[self.data_entry["Image Index"] == image_name]["Finding Labels"].values[0]
        labels = labels.split("|")

        label_vector = torch.zeros(self.get_num_classes(), dtype=torch.float32)
        for label in labels:
            label_vector[self.label_map[label]] = 1.0
        return image, label_vector

    def get_labels(self) -> list[str]:
        """
        Get all the labels in the dataset
        :return: list of labels
        """
        labels = self.data_entry["Finding Labels"].str.split("|")
        all_labels = [label for sublist in labels for label in sublist]
        unique_labels = set(all_labels)
        return list(unique_labels)

    def get_class_distribution(self) -> pd.DataFrame:
        """
        Get the class distribution of the dataset
        :return: dataframe with class distribution
        """
        labels = self.data_entry["Finding Labels"].str.split("|")
        all_labels = [label for sublist in labels for label in sublist]
        label_counts = Counter(all_labels)
        distribution_df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['Count'])
        distribution_df = distribution_df.reset_index().rename(columns={'index': 'Finding Labels'})
        distribution_df = distribution_df.sort_values('Count', ascending=False)
        return distribution_df

    def get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset
        :return: number of classes
        """
        return len(self.get_labels())
