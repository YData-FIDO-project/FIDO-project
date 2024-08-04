"""
Extracting text from images (currently with the help of Pytesseract)
"""

import numpy as np
import pandas as pd
import re
import cv2
import pytesseract
from PIL import Image
from consts_and_weights.labels import CATEGORY_NAME_DICT

CONFIDENCE_THRESHOLD = 2  # from the tutorial


def extracting_text_from_image(img: np.array):
    """
    Using Pytesseract to extract text from provided image.
    Implemented image rotation from this tutorial:
    https://indiantechwarrior.medium.com/optimizing-rotation-accuracy-for-ocr-fbfb785c504b

    :param img: image open as a numpy array

    :returns: text (string)
    """
  
    try:
        # checking if image needs to be rotated
        meta = pytesseract.image_to_osd(img, config=' — psm 0')
        angle = int(re.search(r'Orientation in degrees: \d+', meta).group().split(':')[-1].strip())
        confidence = float(re.search(r'Orientation confidence: \d+', meta).group().split(':')[-1].strip())
        print(f'Orientation: {angle}, confidence: {confidence}')
        # rotating only images with confidence > threshold (default 2):
        if angle == 90 and confidence >= CONFIDENCE_THRESHOLD:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            print('- Image rotated')
        elif angle == 180 and confidence >= CONFIDENCE_THRESHOLD:
            img = cv2.rotate(img, cv2.ROTATE_180)
            print('- Image rotated')
        elif angle == 270 and confidence >= CONFIDENCE_THRESHOLD:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            print('- Image rotated!')

        # extracting text
        text = pytesseract.image_to_string(img)
        if text:  # not an empty string
            return text
        else:
            print("Failed to extract the text")
            return ''

    except Exception as e:
        print(f"Error processing current image: {e}")
        return ''


def image_to_text_pipeline(img_path: str) -> str:
    """
    Text extraction module: from image to text

    :param img_path: path to image stored locally

    :returns: extracted text (str)
    """

    try:
        input_image = Image.open(img_path)
    except Exception as e:
        print(f'Error: {str(e)}. Please review the file manually.')
        return ''

    img_array = np.array(input_image)
    text = extracting_text_from_image(img=img_array)

    return text


def encoding_labels(df: pd.DataFrame, label_dict: dict = CATEGORY_NAME_DICT) -> pd.DataFrame:
    """
    Encoding labels to digits in a dataset

    :param df: df with extracted texts and metadata. Assumes column "category" (str)
    :param label_dict: dictionary digit-to-label

    :returns: df with a column "label" (int)
    """

    # reversing label_dict
    label_to_digit_dict = {v: k for k, v in label_dict.items()}

    df['label'] = df['category'].map(lambda x:
                                     label_to_digit_dict[x]
                                     if x in label_to_digit_dict.keys()
                                     else None)
    print('Label encoding finished')
    print(df['label'].value_counts(dropna=False))

    return df


def combining_all_names(df: pd.DataFrame, method: str = 'first') -> pd.Series:

    """
    This function is not called in the project,
    but can be used for combining all customer names together in a full name

    :param df: df with customer names
    (assumes columns 'user_id', 'firstname', 'middlename' and 'lastname'
    :param method: "first" (returning first instance), "last" (returnin last instance)
    or full (returning a list)

    :returns: pd.Series with the full customer_name (indexed by user_id)
    """

    name_cols = ['firstname', 'middlename', 'lastname']
    df['customer_name'] = (df
                           .apply(lambda x: ' '.join(x[name_cols].dropna()), axis=1)
                           .replace({r'\s+': ' '}, regex=True)
                           .str.upper()
                           )

    if method == 'first':
        names_lookup = df.groupby('user_id')['customer_name'].first()

    elif method == 'last':
        names_lookup = df.groupby('user_id')['customer_name'].last()

    elif method == 'list':
        names_lookup = df.groupby('user_id')['customer_name'].apply(list)

    else:
        return None

    return names_lookup

