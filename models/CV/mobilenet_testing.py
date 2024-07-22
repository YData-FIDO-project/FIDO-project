"""
Code for testing MobileNet model
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.CV.MobileNet_classifier import CVDataset, loading_mobilenet
from consts_and_weights.labels import CATEGORY_NAME_DICT
from utils.evaluation_utils import *

PATH_TO_IMAGES = 'outputs'
PATH_TO_WEIGHTS = 'consts_and_weights/mobilenet_small_all_data_10epochs_with_rejected.pth'
BATCH_SIZE = 16
IMG_SIZE = (256, 256)


def mobilenet_testing(df: pd.DataFrame,
                      path_to_images: str = PATH_TO_IMAGES,
                      path_to_weights: str = PATH_TO_WEIGHTS,
                      batch_size: int = BATCH_SIZE,
                      test_mode: bool = True):
    """
    Classifying a batch of documents. Can be used for processing multiple documents
    (with test_mode = False) or testing retrained model (with test_mode = True)

    :param df: dataframe with metadata texts. Assumes columns "file_name"
    and "label" (digit, can be empty)
    :param path_to_images: path to the folder with images
    :param path_to_weights: path to saved model weights
    :param batch_size: size of the dataloader batch
    :param test_mode: if we are testing the model performance
    (i.e. ground truth label available) or not (i.e. no labels)

    :returns: df with predicted categories and probabilities
    """

    df.reset_index(drop=True, inplace=True)

    # to GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # initializing transforms
    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    # creating the dataset
    test_dataset = CVDataset(path_to_images=path_to_images,
                             df_metadata=df,
                             transform=data_transforms
                             )
    dataset_size = test_dataset.__len__()
    print(f'Dataset size: {dataset_size :,.0f} samples')

    # creating dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=2, prefetch_factor=2,
                                 shuffle=True)
    dataloader_size = len(test_dataloader)
    print(f'Dataloader size: {dataloader_size :,.0f} batches')

    # initializing the model
    model = loading_mobilenet()
    print(f'Downloaded MobileNet')

    # using pretrained weights
    model.load_state_dict(torch.load(path_to_weights, map_location=device))
    model = model.to(device)
    print(f'Downloaded pretrained weights')

    all_predicted_categories = []
    all_predicted_probabilities = []
    all_labels = []
    all_filenames = []

    model.eval()

    # loop for inputs / outputs
    for n, (inputs, labels, names) in enumerate(test_dataloader):
        if n % 10 == 0:
            print(f'- BATCH: {n} / {dataloader_size}')

        inputs = inputs.to(device)
        all_filenames.extend(names)

        if test_mode:
            all_labels.extend(labels)

        with torch.no_grad():
            outputs = model(inputs)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = softmax_outputs.cpu().numpy()
            predicted_categories = probabilities.argmax(1)
            predicted_probabilities = probabilities[np.arange(probabilities.shape[0]),
                                                    predicted_categories]

            all_predicted_categories.extend(predicted_categories)
            all_predicted_probabilities.extend(predicted_probabilities)

    all_predicted_categories = np.array(all_predicted_categories)
    all_predicted_probabilities = np.array(all_predicted_probabilities)
    all_labels = np.array(all_labels)  # empty if not in test mode

    df_results = pd.DataFrame()
    df_results['file_name'] = all_filenames
    df_results['prediction'] = all_predicted_categories
    df_results['proba'] = all_predicted_probabilities
    if test_mode:
        df_results['ground_truth'] = all_labels

        print_main_metrics(df_results['ground_truth'], df_results['prediction'])
        show_classification_report(df_results['ground_truth'],
                                   df_results['prediction'],
                                   labels=CATEGORY_NAME_DICT
                                   )
        plot_confusion_matrix(df_results['ground_truth'],
                              df_results['prediction'],
                              labels=list(CATEGORY_NAME_DICT.values()),
                              normalize='true'  # normalized over true label
                              )

    return df_results
