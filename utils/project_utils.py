"""
Storing all helper functions
"""
from rapidfuzz import fuzz
import re

from consts_and_weights.categories import ALL_CATEGORIES_LIST


def matching_category(input_category: str, predicted_category: str) -> bool:
    """
    Checking if document was uploaded with the right category.

    :param input_category: category under which the document was uploaded
    :param predicted_category: prediction from our model ensemble

    :returns: True / False (bool)
    """

    # preprocessing category names
    input_category, predicted_category = input_category.strip().lower().replace(' ', '_'),\
                                         predicted_category.strip().lower().replace(' ', '_')

    # asserting that all categories are known to the model
    assert input_category in ALL_CATEGORIES_LIST, "Unfamiliar input category"
    assert predicted_category in ALL_CATEGORIES_LIST, "Unfamiliar predicted category"

    return input_category == predicted_category


def matching_name(input_name: str, text: str, threshold: int = 80):
    """
    Checking if uploaded document contains name of the customer

    :param input_name: customer name in the database
    :param text: text extracted from the uploaded document
    :param threshold: threshold value for fuzzywuzz similarity score

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
        if match_score >= threshold:
            total_score += match_score

    total_score /= (100 * len(name_preprocessed))
    found_name = total_score > .6
    print(f'Same customer: {found_name} (confidence level: {total_score :.0%})')
    print(f'Parts of the name recognized: {matches}')

    return found_name, matches, total_score,

