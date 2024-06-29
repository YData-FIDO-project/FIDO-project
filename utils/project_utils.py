"""
Storing all helper functions
"""

from consts_and_weights.categories import ALL_CATEGORIES_LIST

def matching_category(input_category: str, predicted_category: str):
  """
  Checking if document was uploaded with the right category.

  :param input_category: category under which the document was uploaded
  :param predicted_category: prediction from our model ensemble

  :returns: True / False (bool)
  """

  input_category, predicted_category = input_category.strip().lower(), predicted_category.strip().lower()

  assert input_category in ALL_CATEGORIES_LIST, "Unfamiliar input category"
  assert predicted_category in ALL_CATEGORIES_LIST, "Unfamiliar predicted category"
  
  return True if input_category == predicted_category else False

def matching_name(input_name: str, text: str):
  """
  Checking if uploaded document contains name of the customer
  :param input_name: customer name in the database
  :param text: text extracted from the uploaded document
  :returns: True / False (bool)
  """
  
  pass
  
