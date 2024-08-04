"""
Necessary functions for working with PDFs
"""

import PyPDF2
from pdf2image import convert_from_path
import os

from consts_and_weights.scanners import SCANNERS


def converting_pdf_to_jpg(file_path: str, output_dir: str = 'outputs', verbose: bool = True):
    """
    Converting PDF file to image
    :param file_path: PDF file
    :param output_dir: directory for storing converted file

    :returns: list of paths to all converted images (1 PDF page = 1 JPG image), list of filenames
    """
    # creating output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0].lower()
        image = convert_from_path(file_path)
        all_image_paths, all_names = [], []

        for n, img in enumerate(image):
            file_name = f'{base_name}_page_{n + 1}.jpg'
            image_path = os.path.join(output_dir, file_name)
            img.save(image_path, 'JPEG')
            all_image_paths.append(image_path)
            all_names.append(file_name)

        if verbose:
            print('Converted PDF to JPG')
        return all_image_paths, all_names
    except Exception as e:
        if verbose:
            print(f'Failed to convert the document: {e}')
        return [], []


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

        if text in SCANNERS:  # returned only "Scanned by CamScanner" etc.
            print('Failed to extract text from PDF; converting to JPG')
            return ''

        if not text.strip():  # returned empty string
            print('Failed to extract text from PDF; converting to JPG')
            return ''

        return text

    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ''
