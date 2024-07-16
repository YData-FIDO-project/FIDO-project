import matplotlib.pyplot as plt
import seaborn as sns


def plot_convergence(dict_res):

  """
  Plotting loss and accuracy
  """
  fig, axes = plt.subplots(1, 2, figsize=(14,4))
  types=['Loss', 'Accuracy']
  plt.subplot(1,2,1); plt.plot(dict_res['train'][0], color='darkslateblue'); plt.plot(dict_res['val'][0], color='indianred');
  plt.title('Loss'); plt.legend(['train','val'])
  plt.subplot(1,2,2); plt.plot(dict_res['train'][1], color='darkslateblue'); plt.plot(dict_res['val'][1], color='indianred');
  plt.title('Accuracy'); plt.legend(['train','val'])

def plot_confusion_matrix(y_true, y_pred, labels: list, normalize = False):

  """
  Plotting confusion matrix
  :param y_true: ground truth
  :param y_pred: prediction
  :param labels: meaningful label names
  :param normalize: normalize over rows (string 'true') or columns (string 'predicted')
  """


  palette = sns.diverging_palette(230, 20, as_cmap=True)

  if normalize:
    cm = confusion_matrix(y_true, y_preds, normalize=normalize)
  else:
    cm = confusion_matrix(y_true, y_preds)

  format = ',.0f' if not normalize else '.0%'

  plt.figure(figsize=(8, 8))
  sns.heatmap(cm, annot=True, fmt=format, cmap=palette, alpha=.7,
              xticklabels=labels, yticklabels=labels)
  plt.title('Confusion Matrix', fontsize=14)
  plt.xlabel('Predicted Labels', fontsize=12)
  plt.ylabel('True Labels', fontsize=12)
  plt.show()

  # classification report

  cr = classification_report(y_true, y_preds,
                             labels=[k for k in label_name_dict.keys()],
                             target_names=[v for v in label_name_dict.values()],
                             # output_dict=True
                             )
  print(cr)
