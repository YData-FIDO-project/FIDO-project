"""
Main project pipeline

Author: Valeriya Vazhnova
Date: 2024-08-04
"""

import fire
import pandas as pd

from consts_and_weights.categories import ALL_CATEGORIES_LIST
from utils.AWS_utils import downloading_image_from_s3
from utils.text_utils import image_to_text_pipeline
from utils.pdf_utils import *
from utils.random_state_utils import fix_randomization
from models.NLP.bert_inference import bert_inference
from models.CV.mobilenet_inference import mobilenet_inference
from models.ensemble import ensemble_testing
from utils.project_utils import matching_category, matching_name

LOCAL_DIR = 'outputs'


def main(img_uri: str, key_id: str, secret_access_key: str,
         customer_name: str, input_category: str, random_state: int = None,
         local_dir: str = LOCAL_DIR):
    """
    Validating an input document

    :param img_uri:
    :param key_id:
    :param secret_access_key:
    :param customer_name:
    :param input_category:
    :param random_state:
    :param local_dir: path to local directory in which we save the file
    :return:
    """

    # initial checks
    input_category = input_category.lower().replace(' ', '_')  # same format
    if input_category not in ALL_CATEGORIES_LIST:
        print(f'Unfamiliar category: {input_category}. Please review this document manually.')
        return

    # storing metadata
    img_info = {'uri': img_uri, 'input_category': input_category, 'customer_name': customer_name}

    # handling rare categories
    rare_categories = ['form_3', 'property_rate', 'mortgage_statement', 'form_4']
    if input_category in rare_categories:
        answer = input(
            f'Input belongs to a minority category. Manual review suggested. '
            f'Continue with the model prediction (Y/N)?')
        if not answer.strip().lower().startswith('y'):
            return

    # downloading the file from AWS
    local_path_to_image, img_name = downloading_image_from_s3(
        img_uri=img_uri, key_id=key_id,
        secret_access_key=secret_access_key,
        local_dir=local_dir)

    if not local_path_to_image:
        print('Failed to download image.')
        return
    else:
        img_info['local_path'] = local_path_to_image
        img_info['file_name'] = img_name

    # working with images
    if local_path_to_image.strip().lower().endswith(('.jpeg', '.jpg', '.png')):
        text = image_to_text_pipeline(local_path_to_image)
        if not text:
            print('Did not manage to extract text. Please review the file manually.')
            return
        else:
            img_info['text'] = text
            print('Extracted text successfully')

    # working with PDFs
    elif local_path_to_image.strip().lower().endswith('.pdf'):
        # text extraction
        text = extracting_text_from_pdf(local_path_to_image)
        img_info['text'] = text
        print('Text extraction from PDF')

        # converting to JPG
        all_paths_to_converted_image, all_converted_names = converting_pdf_to_jpg(local_path_to_image)
        if all_paths_to_converted_image:
            # rewriting with path to JPG (1st file)
            img_info['local_path'] = all_paths_to_converted_image[0]
            img_info['file_name'] = all_converted_names[0]  # rewriting with JPG name (1st file)

        # if didn't manage to extract text from PDF, using pytesseract
        if not text:
            text_from_jpg = ''
            for p in all_paths_to_converted_image:
                t = image_to_text_pipeline(p)
                text_from_jpg += t+'\n' if t else ''

            if not text_from_jpg:
                print('Did not manage to extract text. Please review the file manually.')
                return
            else:
                img_info['text'] = text_from_jpg
                print('Extracted text successfully')

    # assigning random state if parameter is not null
    if random_state:
        fix_randomization(random_state)
        print(f'Random state: {random_state}')

    # BERT inference
    print('\n--- STARTING NLP INFERENCE ---')
    nlp_category, nlp_proba = bert_inference(document=img_info['text'])
    df_nlp = pd.DataFrame([{'file_name': img_info['file_name'],
                            'prediction': nlp_category,
                            'proba': nlp_proba}])

    # MobileNet inference
    print('\n--- STARTING CV INFERENCE ---')
    cv_category, cv_proba = mobilenet_inference(path_to_image=img_info['local_path'])
    df_cv = pd.DataFrame([{'file_name': img_info['file_name'],
                           'prediction': cv_category,
                           'proba': cv_proba}])

    # ensemble score
    print('\n--- CALCULATING ENSEMBLE SCORE ---')
    df_ensemble = ensemble_testing(df_nlp=df_nlp, df_cv=df_cv)

    print("\n--- FINAL PREDICTION ---")
    print(f'Category: {df_ensemble["prediction_cat_name"][0]}')
    print(f'Probability:\n- CV = {df_ensemble["cv_proba"][0] :,.3f}'
          f'\n- NLP = {df_ensemble["nlp_proba"][0] :,.3f}')

    # category matching
    category_match = matching_category(
        input_category=input_category,
        predicted_category=df_ensemble["prediction_cat_name"][0]
    )
    img_info['category_is_correct'] = category_match
    if category_match:
        print('- Category: correct')

    # name matching
    name_match, name_parts, name_score = matching_name(input_name=customer_name,
                                                       text=img_info['text'])
    img_info['name_is_correct'] = name_match
    img_info['name_confidence'] = name_score
    if name_match:
        print(f'- Name: correct')

    # compile file with results
    df_final = pd.DataFrame([img_info]).merge(df_ensemble, on='file_name')

    # save file with results locally
    final_file_path = os.path.join(local_dir, 'df_results.csv')
    df_final.to_csv(final_file_path, index=False)
    print(f'Results saved: {final_file_path}')


if __name__ == '__main__':
    fire.Fire(main)
