"""
Code for training BERT model
"""

# TODO: check which packages are actually needed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.NLP.BERT_classifier import BertDataset, BERTClassifier
from consts_and_weights.labels import CATEGORY_NAME_DICT

BERT_MODEL_NAME = 'bert-base-uncased'
PATH_TO_WEIGHTS = 'consts_and_weights/bert_13classes_10epochs_adam_full_data_with_rejected.pth'
BATCH_SIZE = 16


def BERT_testing(df: pd.DataFrame,
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
    :param model_name: model nama
    :param label_dict: dictionary with category names (digit to label)
    :param batch_size: size of the dataloader batch
    :param test_mode: if we are testing the model performance
    (i.e. ground truth label available) or not (i.e. no labels)

    :returns: predicted categories, probabilities
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

    # creating a dataset
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

    # TODO: add output of file name...

    model.eval()

    # loop for inputs / outputs
    for n, batch in enumerate(test_dataloader):
        if n % 10 == 0:
            print(f'- BATCH: {n} / {dataloader_size}')

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        if test_mode:
            labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = softmax_outputs.cpu().detach().numpy()[:, 0]

            # TODO: check which axis I need to apply this to
            predicted_category = probabilities.argmax(1)
            # probability = probabilities[predicted_category]

    # print(f'Prediction: {label_dict[predicted_category]} ({probability :,.3f}) ')

    return predicted_category, # probability




  # y_true = np.empty((0, ))
  # y_pred = np.empty((0, ))
  # probabilities = np.empty((0, len(label_name_dict)))
  #
  #
  #
  #
  #   # saving labels
  #   y_true = np.hstack((y_true, labels.cpu().detach().numpy()))
  #
  #   with torch.no_grad():
  #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
  #     outputs = torch.nn.functional.softmax(outputs, dim=1)
  #     probabilities = np.vstack((probabilities, outputs.cpu().detach().numpy()))
  #
  #   # saving predictions
  #   y_pred = np.hstack((y_pred, outputs.cpu().detach().numpy().argmax(1)))
  #
  # return y_true, y_pred, probabilities



# show classification report

# # confusion matrix
# new_labels = [v for v in label_name_dict.values() if v not in ['mortgage_statement', 'property_rate']]
# # labels = [v for v in label_name_dict.values()]
# plot_confusion_matrix(y_true, y_preds,
#                       new_labels
#                       )

# # same, but normalized over rows (true label)
#
# plot_confusion_matrix(y_true, y_preds, new_labels, normalize='true')
