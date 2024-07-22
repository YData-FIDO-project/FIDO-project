"""
Training loop for MobileNet classifier
"""

import pandas as pd
import time
import copy
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from models.CV.MobileNet_classifier import CVDataset, loading_mobilenet
from consts_and_weights.labels import CATEGORY_NAME_DICT
from utils.evaluation_utils import plot_convergence

PATH_TO_IMAGES = 'outputs'
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = (256, 256)


def mobilenet_training(df: pd.DataFrame,
                       train_size: int, validation_size: int,
                       path_to_images: str = PATH_TO_IMAGES,
                       label_dict: dict = CATEGORY_NAME_DICT,
                       batch_size: int = BATCH_SIZE,
                       num_epochs: int = EPOCHS,
                       learning_rate: float = LEARNING_RATE):
    """
    Training MobileNet classifier

    :param df: dataframe with metadata texts. Assumes columns "file_name"
    and "label" (digit)
    :param train_size: size of the train set
    :param validation_size: size of the validation set
    :param path_to_images: path to the folder with images
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
    df_train.reset_index(drop=True, inplace=True)
    print('* * *')
    print(f'\nValidation set: {df_val.shape}')
    print(df_val['label_digit'].value_counts(dropna=False, normalize=True).sort_index())
    df_val.reset_index(drop=True, inplace=True)

    n_classes = len(label_dict)

    # to GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # initializing transforms
    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    # creating datasets
    datasets = {}
    for phase, phase_df in zip(['train', 'val'], [df_train, df_val]):
        datasets[phase] = CVDataset(path_to_images=path_to_images,
                                    df_metadata=phase_df,
                                    transform=data_transforms
                                    )
    print(f'Created datasets')

    # TODO: check if it's used somewhere
    dataset_sizes = {x: datasets[x].__len__() for x in ['train', 'val']}

    # creating dataloaders
    dataloader = {}
    for phase in ['train', 'val']:
        print(f"\n{phase.upper()}: {dataset_sizes[phase] :,.0f} samples")
        dataloader[phase] = DataLoader(datasets[phase], batch_size=batch_size,
                                       num_workers=2, prefetch_factor=2,
                                       shuffle=True)
        print(f"{phase.upper()} dataloader size: {len(dataloader[phase])}")

    # initializing the model
    model = loading_mobilenet(CATEGORY_NAME_DICT)
    print(f'Downloaded MobileNet')

    # initializing training parameters
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    num_epochs = num_epochs
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
            for inputs, labels, _ in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_ft.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # collect statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.detach())

                if phase == 'train':
                    exp_lr_scheduler.step()

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
