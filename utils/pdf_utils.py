"""
Necessary functions for working with PDFs
"""

import PyPDF2
from pdf2image import convert_from_path
import os

from consts_and_weights.scanners import SCANNERS


def converting_pdf_to_jpg(file_path: str, output_dir: str = 'outputs'):
    """
    Converting PDF file to image
    :param file_path: PDF file
    :param output_dir: directory for storing converted file

    :returns: JPG image
    """
    # creating output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # file_name = file_path.split('/')[-1].lower().replace('.pdf', '')
    file_name = os.path.splitext(os.path.basename(file_path))[0].lower()
    image = convert_from_path(file_path)
    all_image_paths = []

    for n, img in enumerate(image):
        file_name = f'{file_name}_page_{n + 1}.jpg'
        image_path = os.path.join(output_dir, file_name)
        img.save(image_path, 'JPEG')
        all_image_paths.append(image_path)

    return all_image_paths

def extracting_text_from_pdf(file_path) -> str:
    """
    Extracting text from PDF file
    :param file_path: path to the PDF file

    :returns: text (str)
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_number in range(len(reader.pages)):
                page = reader.pages[page_number]
                text += page.extract_text() or ""
        if text.strip():  # if not empty string
            return text
        elif text in SCANNERS:  # returned only "Scanned by CamScanner" etc.
            print('Failed to extract text from PDF; converting to JPG')
            return None
        else:
            print('Failed to extract text from PDF; converting to JPG')
            return None
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return None

