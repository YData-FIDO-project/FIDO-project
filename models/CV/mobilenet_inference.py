"""
Inference on MobileNet model
"""
from PIL import Image
import torch
from torchvision import transforms

from models.CV.MobileNet_classifier import loading_mobilenet
from consts_and_weights.labels import CATEGORY_NAME_DICT

PATH_TO_WEIGHTS = '../../consts_and_weights/mobilenet_small_all_data_10epochs_with_rejected.pth'
BATCH_SIZE = 16
IMG_SIZE = (256, 256)


def mobilenet_inference(path_to_image: str,
                        path_to_weights: str = PATH_TO_WEIGHTS,
                        label_dict: dict = CATEGORY_NAME_DICT):
    """
    Classifying provided document

    :param path_to_image: path to image to classify (JPG)
    :param path_to_weights: path to saved model weights
    :param label_dict: dictionary with category names (digit to label)

    :returns: category (digit), softmax score
    """

    # to GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # initializing transforms
    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    # loading and transforming the image
    image = Image.open(path_to_image).convert("RGB")
    inputs = data_transforms(image).unsqueeze(0).to(device)  # batch of 1 image
    print(f'Downloaded and transformed the image')

    # initializing the model
    model = loading_mobilenet()
    print(f'Downloaded MobileNet')

    # using pretrained weights
    model.load_state_dict(torch.load(path_to_weights, map_location=device))
    model = model.to(device)
    print(f'Downloaded pretrained weights')

    model.eval()

    with torch.no_grad():
        outputs = model(inputs)
        softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
        probabilities = softmax_outputs.cpu().detach().numpy()

        predicted_category = probabilities.argmax(1)[0]
        probability = probabilities[0][predicted_category]

    print(f'Prediction: {label_dict[predicted_category]} ({probability :,.3f}) ')

    return predicted_category, probability

