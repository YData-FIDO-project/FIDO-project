"""
Code for training BERT model
"""

# TODO: check which packages are actually needed
import torch
import copy
import time
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd

from models.NLP.BERT_classifier import BertDataset, BERTClassifier
from consts_and_weights.labels import CATEGORY_NAME_DICT
from utils.evaluation_utils import plot_convergence

BERT_MODEL_NAME = 'bert-base-uncased'
PATH_TO_WEIGHTS = 'consts_and_weights/bert_13classes_10epochs_adam_full_data_with_rejected.pth'
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5


def bert_training(df: pd.DataFrame,
                  train_size: int, validation_size: int,
                  model_name: str = BERT_MODEL_NAME,
                  label_dict: dict = CATEGORY_NAME_DICT,
                  batch_size: int = BATCH_SIZE,
                  num_epochs: int = EPOCHS,
                  learning_rate: float = LEARNING_RATE):
    """
    Training BERT classifier

    :param df: dataframe with extracted texts. Assumes columns "text" (document text)
    and "label" (digit)
    :param train_size: size of the train set
    :param validation_size: size of the validation set
    :param model_name: model name
    :param label_dict: dictionary with category names (digit to label)
    :param batch_size: size of the dataloader batch
    :param num_epochs: number of training epochs
    :param learning_rate: learning rate

    :returns: trained model, model weights
    """

    # train / validation split
    df_train, df_val = train_test_split(df, train_size=train_size, test_size=validation_size,
                                        # random_state=42,
                                        stratify=df['label'])
    print(f'Split the data to train and validation set')
    print(f'Train set: {df_train.shape}')
    print(df_train['label_digit'].value_counts(dropna=False, normalize=True).sort_index())
    print('* * *')
    print(f'\nValidation set: {df_val.shape}')
    print(df_val['label_digit'].value_counts(dropna=False, normalize=True).sort_index())

    df.reset_index(drop=True, inplace=True)

    n_classes = len(label_dict)

    # to GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # creating a BERT dataset
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    print(f'Downloaded the tokenizer')

    # creating datasets
    datasets = {}
    for phase, ds in zip(['train', 'val'], [df_train, df_val]):
        datasets[phase] = BertDataset(ds, tokenizer)
    print(f'Created datasets')

    # TODO: check if it's used somewhere
    dataset_sizes = {x: datasets[x].__len__() for x in ['train', 'val']}

    # creating dataloaders
    dataloader = {}
    for phase in ['train', 'val']:
        print(f"\n{phase.upper()}: {dataset_sizes[phase] :,.0f} samples")
        dataloader[phase] = DataLoader(datasets[phase], batch_size=BATCH_SIZE,
                                       num_workers=2, prefetch_factor=2,
                                       shuffle=True)
        print(f"{phase.upper()} dataloader size: {len(dataloader[phase])}")

    # initializing the model
    # TODO: add early stopping
    model = BERTClassifier(bert_model_name=model_name,
                           num_classes=n_classes
                           ).to(device)
    print(f'Downloaded model: {model_name}')

    # initializing training parameters
    training_steps = len(dataloader['train']) * num_epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=training_steps)
    print(f'Training parameters set')

    # starting point
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = .0

    # results
    result_dict = {
        'train': {'loss': [], 'accuracy': []},
        'val': {'loss': [], 'accuracy': []}
    }

    # iterating through epochs
    for e in range(num_epochs):
        print(f'\n\nEPOCH {e + 1} / {num_epochs}')
        print('-' * 12)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterating over data batches
            for batch in dataloader[phase]:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Collect statistics
                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(predictions == labels.detach())

                if phase == 'train':
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_time = time.time() - since

            print(f'{phase.upper()} Loss: {epoch_loss :.4f} Accuracy: {epoch_acc :.4f}')
            if phase == 'val':
                print(f'Time elapsed: {(epoch_time // 60) :.0f} m {(epoch_time % 60) :.0f} s')

            result_dict[phase]['loss'].append(epoch_loss)
            result_dict[phase]['accuracy'].append(epoch_acc)

            # saving best model results
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60) :.0f} m {(time_elapsed % 60) :.0f} s')
    print(f'Best Validation Accuracy: {best_acc :4f}')

    # load best model weights
    model.load_state_dict(best_model_weights)

    # plotting loss and accuracy
    plot_convergence(result_dict)

    return model, result_dict
