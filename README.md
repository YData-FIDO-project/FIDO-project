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
...

### 2. Retraining the model
...


### 3. Adding / removing document categories

Classifier function is trained on the categories from the closed list. Category names are available in <code>consts_and_weights/categories.py</code> and in <code>consts_and_weights/labels.py</code> (digit to label dictionary).

For adding a new category, please add its name to the <code>categories.py</code> file and retrain the models in the <code>models</code> folder:
- MobileNet classifier (file <code>models/CV/mobilenet_training.py</code>)
- BERT classifier (file <code>models/NLP/bert_training.py</code>)

### 4. Changing OCR tool
Current OCR tool implemented in the project is Pytesseract. If you wish to change it, please update the function <code>extracting_text_from_image</code> located in <code>utils/text_extraction.py</code>

## Folder structure

- main.py
- requirements.txt

### outputs
*Default folder for script outputs (downloads and generated csv files).*

### consts_and_weights
*Constants used in the project and weights for pretrained models*

-<code>categories.py</code>: List of categories used in training
-<code>labels.py</code>: Category dictionary (digit to label)
- <code>scanners.py</code>: List of mobile scanner app watermarks *(if text extracted from PDF consists of this watermark only, it will be ignored)*
- <code>bert_13classes_10epochs_adam_full_data_with_rejected.pth</code>: Weights for BERT *(Git LFS pointer)*
- <code>mobilenet_large_all_data_10epochs_with_rejected.pth</code>: Weights for MobileNet *(Git LFS pointer)*

  
### models

**MobileNet models:**
- <code>MobileNet_classifier.py</code>: initializing the classifier
- <code>mobilenet_training.py</code>: code for training/retraining the model
- <code>mobilenet_testing.py</code>: code for testing the model performance
- <code>mobilenet_inference.py</code>: inference for 1 document
- <code>mobilenet_inference_batch.py</code>: inference for several documents *(in case of retroscoring)*

**BERT models:**
- <code>BERT_classifier.py</code>: initializing the classifier
- <code>bert_training.py</code>: code for training/retraining the model
- <code>bert_testing.py</code>: code for testing the model performance
- <code>bert_inference.py</code>: inference for 1 document
- <code>bert_inference_batch.py</code>: inference for several documents *(in case of retroscoring)*

### utils
Project helper functions
- <code>AWS_utils.py</code>: script for downloading a document from AWS S3 bucket
- <code>img_utils.py</code>: ...
- <code>pdf_utils.py</code>: scripts for transforming PDF documents *(e.g. extracting texts, converting to JPG)*
- <code>project_utils.py</code>: scripts for the pipeline *(matching document category and customer name)*
- <code>text_extraction.py</code>: script for OCR

