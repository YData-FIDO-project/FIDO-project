"""
Necessary functions for working with PDFs
"""


def converting_pdf_to_jpg(file):
    """
    Converting PDF file to image
    :param file: PDF file

    :returns: JPG image
    """
    img = ...
    return img


def extracting_text_from_pdf(file):
    """
    Extracting text from PDF file
    :param file: PDF file
    returns: text (str)
    """
    text = ...
    if text:  # if not empty string
        return text
    else:
        print('Failed to extract text from PDF; converting to JPG')
        return None
