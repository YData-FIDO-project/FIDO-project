"""
Extracting text from images (currently with the use of Pytesseract)
"""

# TODO: clean up the imports
# import os
import numpy as np
import pandas as pd
import re
# from PIL import Image
import cv2
import pytesseract
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
            return None

    except Exception as e:
        print(f"Error processing current image: {e}")
        return None


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


def compiling_dataframe() -> pd.DataFrame:
    """
    Compiling a dataframe from input data

    :returns: df with extracted text and image metadata
    """
    pass

def combining_all_names():
    pass
