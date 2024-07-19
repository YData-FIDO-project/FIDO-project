"""
Code for training BERT model
"""

# TODO: check which packages are actually needed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from models.NLP.BERT_classifier import BertDataset, BERTClassifier
from consts_and_weights.labels import CATEGORY_NAME_DICT
from utils.evaluation_utils import *

BERT_MODEL_NAME = 'bert-base-uncased'
PATH_TO_WEIGHTS = 'consts_and_weights/bert_13classes_10epochs_adam_full_data_with_rejected.pth'
BATCH_SIZE = 16


def bert_testing(df: pd.DataFrame,
                 path_to_weights: str = PATH_TO_WEIGHTS,
                 model_name: str = BERT_MODEL_NAME,
                 label_dict: dict = CATEGORY_NAME_DICT,
                 batch_size: int = BATCH_SIZE,
                 test_mode: bool = True):
    """
    Classifying a batch of documents. Can be used for processing multiple documents
    (with test_mode = False) or testing retrained model (with test_mode = True)

    :param df: dataframe with extracted texts. Assumes columns "text" (document text)
    and "label" (can be empty).
    :param path_to_weights: path to saved model weights
    :param model_name: model name
    :param label_dict: dictionary with category names (digit to label)
    :param batch_size: size of the dataloader batch
    :param test_mode: if we are testing the model performance
    (i.e. ground truth label available) or not (i.e. no labels)

    :returns: df with predicted categories and probabilities
    """

    df.reset_index(drop=True, inplace=True)

    n_classes = len(label_dict)

    # to GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # initializing the model
    model = BERTClassifier(bert_model_name=model_name,
                           num_classes=n_classes
                           ).to(device)
    print(f'Downloaded model: {model_name}')

    # using pretrained weights
    model.load_state_dict(torch.load(path_to_weights, map_location=device))
    print(f'Downloaded pretrained weights')

    # creating a BERT dataset
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    print(f'Downloaded the tokenizer')
    test_dataset = BertDataset(df, tokenizer)
    dataset_size = test_dataset.__len__()
    print(f'Dataset size: {dataset_size :,.0f} samples')

    # creating dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=2, prefetch_factor=2,
                                 shuffle=True)
    dataloader_size = len(test_dataloader)
    print(f'Dataloader size: {dataloader_size :,.0f} batches')

    all_predicted_categories = []
    all_predicted_probabilities = []
    all_labels = []
    all_filenames = []

    model.eval()

    # loop for inputs / outputs
    for n, batch in enumerate(test_dataloader):
        if n % 10 == 0:
            print(f'- BATCH: {n} / {dataloader_size}')

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        file_names = batch['file_name']

        if test_mode:
            all_labels.extend(batch['label'])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = softmax_outputs.cpu().numpy()
            predicted_categories = probabilities.argmax(1)
            predicted_probabilities = probabilities[np.arange(probabilities.shape[0]), predicted_categories]

            all_predicted_categories.extend(predicted_categories)
            all_predicted_probabilities.extend(predicted_probabilities)
            all_filenames.extend(file_names)

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

