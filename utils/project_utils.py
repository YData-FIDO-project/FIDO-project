"""
Storing all helper functions
"""
from rapidfuzz import fuzz
import re

from consts_and_weights.categories import *


def matching_category(input_type: str, predicted_category: str) -> bool:
    """
    Checking if document was uploaded with the right category.

    :param input_type: category under which the document was uploaded
    :param predicted_category: prediction from our model ensemble

    :returns: True / False (bool)
    """

    # preprocessing category names
    input_type, predicted_category = input_type.strip().replace(' ', '_'), \
                                     predicted_category.strip().lower().replace(' ', '_')

    # asserting that all categories are known to the model
    assert input_type in ['POA', 'POE']
    assert predicted_category in ALL_CATEGORIES_LIST, "Unfamiliar predicted category"

    return predicted_category in CATEGORIES_AND_TYPES_DICT[input_type]


def matching_name(input_name: str, text: str, threshold: float = .65, verbose: bool = True):
    """
    Checking if uploaded document contains name of the customer

    :param input_name: customer name in the database
    :param text: text extracted from the uploaded document
    :param threshold: threshold value for fuzzywuzz similarity score
    :param verbose: if the function should print out the results of the matching (default True)

    :returns: matches (dictionary with name parts and similarity score), total matching score
    """

    # for empty input
    if not input_name or not text:
        print('No name or document text provided')
        return {}, 0.0

    # preprocessing the name
    name_preprocessed = re.sub(r'\s*-\s*', ' ', input_name.strip().upper())
    name_preprocessed = name_preprocessed.split()

    # preprocessing the text
    text = re.sub(r'[^\w\s\d]+', '', text.upper())

    matches = {}
    total_score = 0

    for name_part in name_preprocessed:
        match_score = fuzz.partial_ratio(name_part, text)
        matches[name_part] = round(match_score, 2)
        total_score += match_score

    total_score /= (100 * len(name_preprocessed))
    found_name = total_score > threshold
    if verbose:
        print(f'Same customer: {found_name} (confidence level for recognized parts: {total_score :.0%})')
        print(f'Parts of the name recognized: {matches}')

    return found_name, matches, total_score

