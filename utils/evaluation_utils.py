"""
Main evaluation metrics
"""

from sklearn.metrics import confusion_matrix, classification_report, \
  accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


def print_main_metrics(ground_truth, predictions):
    """
    Printing out main metrics
    :param ground_truth: ground truth values
    :param predictions: predicted values

    :returns: accuracy, precision, recall, f1 score, roc auc score
    """
    print(f'Main metrics:')
    print(f'- Accuracy: {accuracy_score(ground_truth, predictions) :,.3f}')
    print(f'- Precision: {precision_score(ground_truth, predictions, average="weighted") :,.3f}')
    print(f'- Recall: {recall_score(ground_truth, predictions, average="weighted") :,.3f}')
    print(f'- F1 score: {f1_score(ground_truth, predictions, average="weighted") :,.3f}')


def show_classification_report(ground_truth, predictions, labels: dict):
    """
    Printing classification report

    :param ground_truth: ground truth values
    :param predictions: predicted values
    :param labels: dictionary label digit to meaningful name

    :returns: classification report
    """
    cr = classification_report(ground_truth, predictions,
                               labels=list(labels.keys()),
                               target_names=list(labels.values()),
                               )
    print(cr)


def plot_confusion_matrix(ground_truth, predictions,
                          labels: list, normalize=False,
                          save_path: str = None):
    """
    Plotting confusion matrix

    :param ground_truth: ground truth values
    :param predictions: predicted values
    :param labels: meaningful label names
    :param normalize: normalize over rows (string 'true'),
    columns (string 'predicted') or nothing (False)
    :param save_path: path to save the file

    :returns: confusion matrix plot
    """

    palette = sns.diverging_palette(230, 20, as_cmap=True)

    if normalize:
        cm = confusion_matrix(ground_truth, predictions, normalize=normalize)
    else:
        cm = confusion_matrix(ground_truth, predictions)

    value_format = ',.0f' if not normalize else '.0%'

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=value_format, cmap=palette, alpha=.7,
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    if save_path:
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), bbox_inches='tight')

    plt.show()


def plot_convergence(dict_res: dict,
                     second_line: bool = True,
                     save_path: str = 'outputs'):
    """
    Line plot. Loss and accuracy for training loop
    :param dict_res: results (dictionary)
    :param second_line: if there is second line on the plot, default True
    :param save_path: path to save the file

    :returns: convergence plots
    """

    dict_keys = list(dict_res.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 1st subplot (loss)
    axes[0].plot(dict_res[dict_keys[0]]['loss'], color='darkslateblue', label=dict_keys[0])
    if second_line:
        axes[0].plot(dict_res[dict_keys[1]]['loss'], color='indianred', label=dict_keys[1])
    axes[0].set_title('Loss')
    axes[0].legend()

    # 2nd subplot (accuracy)
    axes[1].plot(dict_res[dict_keys[0]]['accuracy'], color='darkslateblue', label=dict_keys[0])
    if second_line:
        axes[1].plot(dict_res[dict_keys[1]]['accuracy'], color='indianred', label=dict_keys[1])
    axes[1].set_title('Accuracy')
    axes[1].legend()

    if save_path:
        plt.savefig(os.path.join(save_path, 'convergence_graphs.png'), bbox_inches='tight')

    plt.show()
