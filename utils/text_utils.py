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
from utils.pdf_utils import converting_pdf_to_jpg, extracting_text_from_pdf

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

    :param df: df with extracted texts and metadata. Assumes column "input_category" (str)
    :param label_dict: dictionary digit-to-label

    :returns: df with a column "label" (int)
    """

    # reversing label_dict
    label_to_digit_dict = {v: k for k, v in label_dict.items()}

    df['label'] = df['input_category'].map(lambda x:
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
        return pd.Series()

    return names_lookup


def ocr_for_a_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracting texts from a batch of downloaded documents

    :param df: df with document metadata; assumes columns "local_path" and "file_name";
    if testing the model, should also contain "input_category" (str)

    :return: same df with extracted texts in 'text' column;
    if input_category (str) exists, it's encoded as a label digit
    """

    # splitting to images and PDFs
    df_images = df[df['local_path'].strip().lower().endswith(('.jpeg', '.jpg', '.png'))]
    df_pdfs = df[df['local_path'].strip().lower().endswith('.pdf')]
    assert df_pdfs.shape[0] + df_images.shape[0] == df.shape[0], "Unknown file format"
    print('Separated images from PDFs')

    # working with images
    df_images['text'] = df['local_path'].map(lambda x: image_to_text_pipeline(x))
    print('Extracted text from images')

    # working with PDFs
    df_pdfs['text'] = df_pdfs['local_path'].map(lambda x: extracting_text_from_pdf(x))
    print('Extracted text from pdfs')

    # converting PDFs to JPG
    # wrapper function
    def apply_converting_pdf_to_jpg(row):
        all_paths_to_converted, all_converted_names = converting_pdf_to_jpg(
            row['local_path'], verbose=False)
        local_path = all_paths_to_converted[0] if all_paths_to_converted else None
        file_name = all_converted_names[0] if all_converted_names else None
        return pd.Series({'local_path': local_path, 'file_name': file_name})

    df_pdfs[['local_path', 'file_name']] = df_pdfs.apply(apply_converting_pdf_to_jpg, axis=1)
    print('Converted PDFs to JPGs')

    # if didn't manage to extract text from PDF, using pytesseract
    df_pdf_no_text = df_pdfs.loc[(df_pdfs['text'] == '') or (df_pdfs['text'].isnull())]
    df_pdfs = df_pdfs.loc[(df_pdfs['text'] != '') and (df_pdfs['text'].notnull())]
    df_pdf_no_text['text'] = df_pdf_no_text['local_path'].map(lambda x: image_to_text_pipeline(x))
    print('Extracted text from converted images')

    # combining the dataframe
    df_final = pd.concat([df_images, df_pdfs, df_pdf_no_text]).reset_index(drop=True)

    if 'input_category' in df_final.columns:
        df_final = encoding_labels(df_final)

    return df_final
