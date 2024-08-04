"""
Running inference on a batch of documents (e.g. for retroscoring the model)

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
from models.NLP.bert_testing import bert_testing
from models.CV.mobilenet_testing import mobilenet_testing
from models.ensemble import ensemble_testing
from utils.project_utils import matching_category, matching_name
from utils.evaluation_utils import *
from consts_and_weights.labels import CATEGORY_NAME_DICT

LOCAL_DIR = 'outputs'


def main(df: pd.DataFrame, key_id: str, secret_access_key: str,
         test_mode: bool = False, label_dict: dict = CATEGORY_NAME_DICT,
         random_state: int = None, local_dir: str = LOCAL_DIR):
    """
    Validating a batch of input documents

    :param df: df with image metadata; assumes columns: 'uri', 'local_path', 'file_name',
    'text' (with extracted text), 'customer_name' (customer name). If test_mode, also 'label'
    (with label as a digit)
    :param key_id: key id to access AWS
    :param secret_access_key: secret access key to access AWS
    :param test_mode: if we are testing the model performance
    (i.e. ground truth label available) or not (i.e. no labels)
    :param label_dict: dictionary with category names (digit to label)
    :param random_state: random state seed if needed (default None)
    :param local_dir: path to local directory in which we save the file

    :returns:
    1. df with full image info + extracted text + model ensemble prediction.
    2. In case we decided to leave rare categories for manual labeling, returns df with rare categories
    3. In case there were entries in the df where we failed to download image (no local_path),
    returns a separate df for such cases
    """

    # handling images that weren't downloaded
    df_failed = df[df['local_path'].isnull()]
    df = df[df['local_path'].notnull()]

    # handling rare categories
    rare_categories = ['form_3', 'property_rate', 'mortgage_statement', 'form_4']
    answer = input('Remove rare categories for manual labeling (Y/N)?')
    if answer.strip().lower().startswith('y'):
        df_rare = df[df['ground_truth'].isin(rare_categories)]
        df = df[~df['ground_truth'].isin(rare_categories)]
        print(f'New df dimensions: {df.shape}')
    else:
        df_rare = pd.DataFrame()

    # assigning random state if parameter is not null
    if random_state:
        fix_randomization(random_state)
        print(f'Random state: {random_state}')

    # BERT inference
    print('\n--- STARTING NLP INFERENCE ---')
    df_nlp = bert_testing(df=df, test_mode=test_mode)

    # MobileNet inference
    print('\n--- STARTING CV INFERENCE ---')
    df_cv = mobilenet_testing(df=df, test_mode=test_mode)

    # ensemble score
    print('\n--- CALCULATING ENSEMBLE SCORE ---')
    df_ensemble = ensemble_testing(df_nlp=df_nlp, df_cv=df_cv, test_mode=test_mode)

    # TODO: add stuff here
    # category matching
    category_match = matching_category(
        input_category=input_category,
        predicted_category=df_ensemble["prediction_cat_name"][0]
    )
    img_info['category_is_correct'] = category_match
    if category_match:
        print('\n- Category: correct')
    else:
        print('\n- ATTENTION: category mismatch')

    # name matching
    name_match, name_parts, name_score = matching_name(input_name=customer_name,
                                                       text=img_info['text'])
    img_info['name_is_correct'] = name_match
    img_info['name_confidence'] = name_score
    if name_match:
        print(f'\n- Name: correct')
    else:
        print('\n- ATTENTION: name mismatch')

    # compile file with results
    df_final = pd.DataFrame([img_info]).merge(df_ensemble, on='file_name')

    # save file with results locally
    final_file_path = os.path.join(local_dir, 'df_results.csv')
    df_final.to_csv(final_file_path, index=False)
    print(f'Results saved: {final_file_path}')

    return df_final, df_rare, df_failed


if __name__ == '__main__':
    fire.Fire(main)