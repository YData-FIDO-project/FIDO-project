"""
Placeholder for the main file with the project pipeline
"""

import fire
import os
import numpy as np
import pandas as pd

from utils.project_utils import matching_category, matching_name
from utils.text_extraction import extracting_text_from_image
from consts_and_weights.categories import ALL_CATEGORIES_LIST


def main(file, name: str, category: str):
    """Validating the input document

    :param file: file uploaded by the customer
    :param name: customer name in the database
    :param category: category under which the document was uploaded in the app

    :returns: document text, predicted category, prediction softmax score, category match (bool), name match (bool)
    """

    assert category in ALL_CATEGORIES_LIST, "Unknown category; manual labeling required"

    pass


if __name__ == '__main__':
    fire.Fire(main)
