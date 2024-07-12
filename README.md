# FIDO-project
Y-Data industry project. 
Document classification and validation for <a href="https://gh.fido.money/">FIDO Credit</a>

<b>Folder structure:</b>

- <code>consts_and_weights</code>: Constants used in the project and weights for pretrained models
- <code>models</code>: Code for retraining models and running inference (MobileNet + BERT)
- <code>utils</code>: Project helper functions

## Validating a document
For validating an input document, please run the <code>main.py</code> file.


## Document categories

Classifier function is trained on the categories from the closed list. Category names are available in <code>consts_and_weights/categories.py</code>. 

For adding a new category, please add its name to the <code>categories.py</code> file and retrain the model in <code>...</code>.