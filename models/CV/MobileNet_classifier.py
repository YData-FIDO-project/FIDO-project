"""
Initializing MobileNet Classifier model
"""

import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models


from consts_and_weights.labels import CATEGORY_NAME_DICT


def loading_mobilenet(label_dict: dict = CATEGORY_NAME_DICT):
    """
    Loading pretrained MobileNet Model. Adjusting the last layer for fine-tuning.

    :param label_dict: dictionary with category names (digit to label)

    :returns: MobileNet classifier with pretrained weights
    """

    n_classes = len(label_dict)

    model = models.mobilenet_v3_small(pretrained=True)

    # freezing the weights
    for param in model.parameters():
        param.requires_grad = False

    # adjusting last layer
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features=in_features, out_features=n_classes)

    # unfreezing last layer for fine-tuning
    for param in model.classifier[3].parameters():
        param.requires_grad = True

    return model


# building the dataset
class CVDataset(Dataset):
    def __init__(self, path_to_images, df_metadata, transform=None):
        self.path = path_to_images
        self.df = df_metadata
        self.transform = transform
        self.images = sorted([f for f in os.listdir(self.path) if f in self.df['file_name']])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.path, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.df.at[self.df['file_name'] == img_name, 'label']

        return image, label, img_name
