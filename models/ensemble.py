"""
Model ensemble code
"""

import pandas as pd

from consts_and_weights.labels import CATEGORY_NAME_DICT
from utils.evaluation_utils import *


def ensemble_testing(df_cv: pd.DataFrame, df_nlp: pd.DataFrame,
                     label_dict: dict = CATEGORY_NAME_DICT,
                     test_mode: bool = False) -> pd.DataFrame:
    """
    Calculating prediction for an ensemble of CV and NLP models.
    In test mode (test_mode=True) comparing it with the ground truth.
    :param df_cv: df with CV model predictions (assumes columns "file_name", "prediction" and "proba")
    :param df_nlp: df with NLP model predictions (assumes columns "file_name", "prediction" and "proba")
    :param label_dict: dictionary with category names (digit to label)
    :param test_mode: if we are testing the model performance
    (i.e. ground truth label available) or not (i.e. no labels)

    :returns: df with ensemble prediction
    """

    # checking that all categories are known
    assert df_cv[(~df_cv['prediction'].isin(label_dict.keys()))
                 & (df_cv['prediction'].notnull())].empty, "Unknown categories in CV model predictions"
    assert df_nlp[(~df_nlp['prediction'].isin(label_dict.keys()))
                  & (df_nlp['prediction'].notnull())].empty, "Unknown categories in NLP model predictions"

    # preprocessing & merging dataframes
    df_cv.columns = ['file_name'] + ['cv_' + col for col in df_cv.columns[1:]]
    df_nlp.columns = ['file_name'] + ['nlp_' + col for col in df_nlp.columns[1:]]
    df = df_cv.merge(df_nlp, on='file_name', how='outer')
    print(f'Samples received by ensemble model: {df.shape[0]}')

    if test_mode:
        assert df.dropna()[
            df.dropna()['cv_ground_truth'] != df.dropna()['nlp_ground_truth']
        ].empty, "Different ground truth labels for the same document"

        # keeping one ground truth column
        df['ground_truth'] = df['cv_ground_truth'].combine_first(df['nlp_ground_truth'])
        df.drop(columns=['cv_ground_truth', 'nlp_ground_truth'], inplace=True)

    # this line covers four cases:
    # 1. both models agree
    # 2. only one model prediction is available
    # 3. no predictions available -> null
    # 4. using NLP as a default prediction (except for cases specified below)
    df['final_prediction'] = df['nlp_prediction'].combine_first(df['cv_prediction'])

    # NLP score very low -> accept CV
    df.loc[(df['cv_prediction'] != df['nlp_prediction'])
           & (df['cv_proba'].ge(.6))
           & (df['nlp_proba'].le(.4)),
           'final_prediction'] = df.loc[(df['cv_prediction'] != df['nlp_prediction'])
                                        & (df['cv_proba'].ge(.6))
                                        & (df['nlp_proba'].le(.4)),
                                        'cv_prediction']

    # CV prediction: water_bill, electricity_bill -> accept CV
    df.loc[(df['cv_prediction'] != df['nlp_prediction'])
           & (df['cv_proba'].ge(.6))
           & (df['nlp_proba'].le(.8))
           & (df['cv_prediction'].isin([3, 13])),
           'final_prediction'] = df.loc[(df['cv_prediction'] != df['nlp_prediction'])
                                        & (df['cv_proba'].ge(.6))
                                        & (df['nlp_proba'].le(.8))
                                        & (df['cv_prediction'].isin([3, 13])),
                                        'cv_prediction']
    print('Final ensemble score calculated')

    # adding readable labels
    df['prediction_cat_name'] = df['final_prediction'].map(lambda x:
                                                           label_dict[x] if pd.notnull(x)
                                                           else None)
    print('Added category names')

    if test_mode:
        # adding readable ground truth labels
        df['ground_truth_cat_name'] = df['ground_truth'].map(lambda x:
                                                             label_dict[x] if pd.notnull(x)
                                                             else None)

        print_main_metrics(df['ground_truth'], df['final_prediction'])
        show_classification_report(df['ground_truth'],
                                   df['final_prediction'],
                                   labels=label_dict
                                   )
        plot_confusion_matrix(df['ground_truth'],
                              df['final_prediction'],
                              labels=list(label_dict.values()),
                              normalize='true'  # normalized over true label
                              )

    return df
