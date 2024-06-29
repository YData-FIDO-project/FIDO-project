"""
Extracting text from images (currently with the use of Pytesseract)
"""

# TODO: clean up the imports
import os
import numpy as np
import pandas as pd
import re
# from PIL import Image
import cv2
import pytesseract

CONFIDENCE_THRESHOLD = 2  # from the tutorial

def extracting_text_from_image(img: np.array):
    """
    Using Pytesseract to extract text from provided image.
    Implemented image rotation from this tutorial:
    https://indiantechwarrior.medium.com/optimizing-rotation-accuracy-for-ocr-fbfb785c504b

    :param img: image open as a numpy array

    returns: text (string)
    """
  
    try:
        # checking if image needs to be rotated
        meta = pytesseract.image_to_osd(img, config=' — psm 0')
        angle = int(re.search(r'Orientation in degrees: \d+', meta).group().split(':')[-1].strip())
        confidence = float(re.search(r'Orientation confidence: \d+', meta).group().split(':')[-1].strip())
        # rotating only images with confidence > threshold (default 2):
        if angle == 90 and confidence > CONFIDENCE_THRESHOLD:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            print('- Image rotated')
        elif angle == 180 and confidence > CONFIDENCE_THRESHOLD:
            img = cv2.rotate(img, cv2.ROTATE_180)
            print('- Image rotated')
        elif angle == 270 and confidence > CONFIDENCE_THRESHOLD:
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
        print(f"Error processing image {img}: {e}")
        return None
