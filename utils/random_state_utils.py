"""
Fixing random state
"""

import random
import numpy as np
import torch


def fix_randomization(n: int = 42):
    """
    Function to fix random state
    :param n: random state (default 42)
    """

    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(n)
        torch.cuda.manual_seed_all(n)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
