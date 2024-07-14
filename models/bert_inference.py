"""
Inference on BERT model
"""
import torch
from transformers import BertTokenizer

from models.BERT_classifier import BERTClassifier
from consts_and_weights.labels import CATEGORY_NAME_DICT


BERT_MODEL_NAME = 'bert-base-uncased'
n_classes = len(CATEGORY_NAME_DICT)

# to GPU
device = torch.device('mps' if torch.backends.mps.is_available() else
                      torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Device: {device}")

# initializing the model
model = BERTClassifier(bert_model_name=BERT_MODEL_NAME,
                       num_classes=n_classes
                       ).to(device)

# using pretrained weights
model.load_state_dict(torch.load(
  'consts_and_weights/bert_13classes_10epochs_adam_full_data_with_rejected.pth'))

# initializing the tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


def inference(clf_model, model_tokenizer, document: str, label_dict: dict = CATEGORY_NAME_DICT):
    """
    Classifying provided document

    :param clf_model: pretrained classifier
    :param model_tokenizer: suitable tokenizer for the model
    :param document: document to classify
    :param label_dict: dictionary with category names (digit to label)

    :returns: category (digit), softmax score
    """

    clf_model.eval()

    # tokenizing input document
    inputs = model_tokenizer(document,
                             return_tensors='pt', truncation=True,
                             padding=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = clf_model(input_ids=input_ids, attention_mask=attention_mask)
        softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
        probabilities = softmax_outputs.cpu().detach().numpy()[0]

        pred_category = probabilities.argmax()
        probability = probabilities[pred_category]

    print(f'Prediction: {label_dict[pred_category]} ({probability :,.3f}) ')
    return pred_category, probability

