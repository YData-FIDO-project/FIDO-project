"""
Code for training BERT model
"""

# TODO: check which packages are actually needed
import os
import torch
import random
import copy
import time

from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_SIZE = 4000
VAL_SIZE = 1157
TEST_SIZE = 1289

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 10
