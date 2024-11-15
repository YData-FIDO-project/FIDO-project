# FIDO-project
Y-Data industry project. 
Document classification and validation for <a href="https://gh.fido.money/">FIDO Credit</a>


## Quickstart: Validating a Document

For validating an input document, please run the <code>main.py</code> file. It takes input parameters:

```
--img_uri="1111a1a1a11a1a1aaaaa01a/POE/11aa111a-1aaa-111a-aa0a-a111aa1a11aa/0.jpg"  # document URI
--key_id="..."  # access to AWS
--secret_access_key="..."  # access to AWS
--customer_name="John Smith"  # name of the customer in the app
--input_category="appointment_letter"  # category under which the file was uploaded
--random_state=None  # random state seed, default None
--local_dir="outputs"  # category where the script will save the downloaded file and the prediction
```

If document belongs to one of the minority categories, the script will suggest using manual labeling instead: <code>Input belongs to a minority category. Manual review suggested. Continue with the model prediction (Y/N)?</code>. Options:

- <code>Y</code>: Continue running current script and get the prediction.
- <code>N</code>: Stop the script.

Output is a csv file saved in <code>local_dir</code>, containing the following columns:

- <code>uri</code>: Link to the document in S3 bucket *(str, from input parameters)*
- <code>input_category</code>: Category under which the document was uploaded *(str, from input parameters)*
- <code>customer_name</code>: Name of the customer registered in the app *(str, from input parameters)*
- <code>local_path</code>: Path to the local folder where the file and the prediction df was saved *(str)*
- <code>file_name</code>: Name under which the file was saved locally *(str)*
- <code>text</code>: Text extracted from the document *(str)*
- <code>category_is_correct</code>: If predicted category matched the upload category *(bool)*
- <code>name_is_correct</code>: If customer name was found in the document *(bool)*
- <code>name_confidence</code>: Level of confidence for the name matching *(float, range 0-1)*
- <code>cv_prediction</code>: Category predicted by MobileNet model *(digit, can be looked up in <code>consts_and_weights/labels.py</code>)*
- <code>cv_proba</code>: Softmax score from MobileNet model *(float, range 0-1)*
- <code>nlp_prediction</code>: Category predicted by BERT *(digit, can be looked up in <code>consts_and_weights/labels.py</code>)*
- <code>nlp_proba</code>: Softmax score from BERT *(float, range 0-1)*
- <code>final_prediction</code>: Prediction made by ensemble of MobileNet and BERT together *(digit, can be looked up in <code>consts_and_weights/labels.py</code>)*
- <code>prediction_cat_name</code>: Prediction category *(str)*

## Main Tasks

### 1. Retroscoring
In order to run inference on a batch of samples / do retroscoring, please follow these steps:

1. Prepare a csv file which includes columns <code>uri</code>, <code>customer_name</code> and <code>input_category</code> (if testing the model)
2. Run <code>downloading_batch_of_images</code> from <code>utils.AWS_utils.py</code> in order to download the files from S3 bucket and save them locally.
3. Run <code>ocr_for_a_batch</code> from <code>utils.text_utils.py</code> in order to extract texts from the images/pdfs. This is a slow function, rough estimate is 5-6 sec/document.
4. Run <code>retroscoring.py</code> (root directory) to get predictions for a batch of documents.

  - If parameter <code>test_mode = True</code>, it will expect column <code>label</code> (digit) with the ground truth and it will calculate main metrics (accuracy, precision, recall, f1 score). It will also output confusion matrix.
  - If parameter <code>test_mode = False</code> (default), it will output only df with prediction results.

### 2. Retraining the model
If you wish to retrain the model on new data, you can find train functions in these files:

- <code>models/NLP/bert_training.py</code> for BERT
- <code>models/CV/mobilenet_training.py</code> for MobileNet
- <code>models/ensemble.py</code> for the model ensemble

In order to prepare training data, please follow the steps 1-3 from p. 1 (retroscoring).

In order to test the models, please run the following files with parameter <code>test_mode = True</code>:

- <code>models/NLP/bert_testing.py</code> for BERT
- <code>models/CV/mobilenet_testing.py</code> for MobileNet 

### 3. Adding / removing document categories

Classifier function is trained on the categories from the closed list. Category names are available in <code>consts_and_weights/categories.py</code> and in <code>consts_and_weights/labels.py</code> (digit to label dictionary).

For adding a new category, please add its name to the <code>categories.py</code> file and retrain the models in the <code>models</code> folder:
- MobileNet classifier (file <code>models/CV/mobilenet_training.py</code>)
- BERT classifier (file <code>models/NLP/bert_training.py</code>)

### 4. Changing OCR tool
Current OCR tool implemented in the project is Pytesseract. If you wish to change it, please update the function <code>extracting_text_from_image</code> located in <code>utils/text_extraction.py</code>

## Folder structure

- <code>main.py</code>
- <code>retroscoring.py</code>
- <code>requirements.txt</code>

### outputs
*Default folder for script outputs (downloads and generated csv files).*

### consts_and_weights
*Constants used in the project and weights for pretrained models*

- <code>categories.py</code>: List of categories used in training
- <code>labels.py</code>: Category dictionary (digit to label)
- <code>scanners.py</code>: List of mobile scanner app watermarks *(if text extracted from PDF consists of this watermark only, it will be ignored)*
- <code>bert_13classes_10epochs_adam_full_data_with_rejected.pth</code>: Weights for BERT *(Git LFS pointer)*
- <code>mobilenet_large_all_data_10epochs_with_rejected.pth</code>: Weights for MobileNet *(Git LFS pointer)*

  
### models

- <code>ensemble.py</code>: Calculating model ensemble prediction (for 1 / several documents)

**MobileNet models:**
- <code>MobileNet_classifier.py</code>: Initializing the classifier
- <code>mobilenet_training.py</code>: Code for training/retraining the model
- <code>mobilenet_testing.py</code>: Code for testing the model performance / inference for several documents *(in case of retroscoring)*
- <code>mobilenet_inference.py</code>: Inference for 1 document


**BERT models:**
- <code>BERT_classifier.py</code>: Initializing the classifier
- <code>bert_training.py</code>: Code for training/retraining the model
- <code>bert_testing.py</code>: Code for testing the model performance / inference for several documents *(in case of retroscoring)*
- <code>bert_inference.py</code>: Inference for 1 document

### utils
Project helper functions
- <code>AWS_utils.py</code>: Script for downloading a document from AWS S3 bucket. Called from <code>main.py</code>
- <code>evaluation_utils.py</code>: Script for printing main metrics (accuracy, f1 score etc), classification report and plotting confusion matrix. These functions are called from <code>models/NLP/bert_testing.py</code>, <code>models/CV/mobilenet_testing.py</code> and <code>models/ensemble.py</code>
- <code>pdf_utils.py</code>: Scripts for transforming PDF documents *(e.g. extracting texts, converting to JPG)*. Called from <code>main.py</code>
- <code>project_utils.py</code>: Scripts for the pipeline *(matching document category and customer name)*. Called from <code>main.py</code>
- <code>random_state_utils.py</code>: Script for assigning random seed. Called from <code>main.py</code>
- <code>text_utils.py</code>: Scripts for OCR, encoding labels and other text-related tasks. Called from <code>main.py</code>

