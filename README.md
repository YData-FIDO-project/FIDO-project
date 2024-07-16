# FIDO-project
Y-Data industry project. 
Document classification and validation for <a href="https://gh.fido.money/">FIDO Credit</a>


## Quickstart: Main Tasks

### 1. Validating a document
For validating an input document, please run the <code>main.py</code> file. It takes as an input:
- URI of the document (AWS S3 URI link)
- customer name in the app
- upload category (i.e. category under which the document was uploaded).

### 2. Retroscoring
...

### 3. Retraining the model
...


### 4. Adding / removing document categories

Classifier function is trained on the categories from the closed list. Category names are available in <code>consts_and_weights/categories.py</code>. 

For adding a new category, please add its name to the <code>categories.py</code> file and retrain the models in the <code>models</code> folder:
- MobileNet classifier (file <code>MobileNet... .py</code>)
- BERT classifier (file <code>BERT... .py</code>)

### 5. Changing OCR tool
Current OCR tool implemented in the project is Pytesseract. If you wish to change it, please update the file <code>utils/text_extraction.py</code>

## Folder structure

- main.py
- requirements.txt

### consts_and_weights
Constants used in the project and weights for pretrained models
- ...

  
### models
**Binary classifiers:**
- TBD

**MobileNet models:**
- <code>MobileNet_classifier.py</code>: initializing the classifier
- <code>mobilenet_training.py</code>: code for training/retraining the model
- <code>mobilenet_testing.py</code>: code for testing the model performance
- <code>mobilenet_inference.py</code>: inference for 1 document
- <code>mobilenet_inference_batch.py</code>: inference for several documents (in case of retroscoring)

**BERT models:**
- <code>BERT_classifier.py</code>: initializing the classifier
- <code>bert_training.py</code>: code for training/retraining the model
- <code>bert_testing.py</code>: code for testing the model performance
- <code>bert_inference.py</code>: inference for 1 document
- <code>bert_inference_batch.py</code>: inference for several documents (in case of retroscoring)

### utils
Project helper functions
- <code>AWS_utils.py</code>: script for downloading a document from AWS S3 bucket
- <code>img_utils.py</code>: ...
- <code>pdf_utils.py</code>: scripts for transforming PDF documents (e.g. extracting texts, converting to JPG)
- <code>project_utils.py</code>: scripts for the pipeline (matching document category and customer name)
- <code>text_extraction.py</code>: script for OCR

