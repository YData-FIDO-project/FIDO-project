"""
Inference on pretrained MobileNet Small
"""

# check which packages are actually needed there
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from __future__ import print_function, division

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import copy
plt.ion()   # interactive mode
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, models, transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
