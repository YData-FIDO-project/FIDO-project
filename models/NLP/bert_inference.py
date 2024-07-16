"""
Inference on BERT model
"""
import torch
from transformers import BertTokenizer

from models.NLP.BERT_classifier import BERTClassifier
from consts_and_weights.labels import CATEGORY_NAME_DICT

BERT_MODEL_NAME = 'bert-base-uncased'
PATH_TO_WEIGHTS = 'consts_and_weights/bert_13classes_10epochs_adam_full_data_with_rejected.pth'


def BERT_inference(document: str, path_to_weights: str = PATH_TO_WEIGHTS,
                   model_name: str = BERT_MODEL_NAME,
                   label_dict: dict = CATEGORY_NAME_DICT):
    """
    Classifying provided document

    :param document: document to classify (string)
    :param model_name: model name
    :param path_to_weights: path to saved model weights
    :param label_dict: dictionary with category names (digit to label)

    :returns: category (digit), softmax score
    """

    n_classes = len(label_dict)

    # to GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # initializing the model
    model = BERTClassifier(bert_model_name=model_name,
                           num_classes=n_classes
                       ).to(device)
    print(f'Downloaded model: {model_name}')

    # using pretrained weights
    model.load_state_dict(torch.load(path_to_weights, map_location=device))
    print(f'Downloaded pretrained weights')

    # initializing the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f'Initialized the tokenizer for: {model_name}')

    model.eval()

    # tokenizing input document
    inputs = tokenizer(document,
                       return_tensors='pt', truncation=True,
                       padding=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    print(f'Tokenized input file')

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
        probabilities = softmax_outputs.cpu().detach().numpy()[0]

        predicted_category = probabilities.argmax()
        probability = probabilities[predicted_category]

    print(f'Prediction: {label_dict[predicted_category]} ({probability :,.3f}) ')

    return predicted_category, probability
