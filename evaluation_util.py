import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

import random

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


# plot diagnostic learning curves
def summarize_metric(history, metric):
    plt.figure(figsize=(6, 4))
    plt.title('Baseline Model Training and Validation Loss')
    plt.plot(range(1,len(history.history[metric]) + 1),history.history[metric], color='red', label=f'Train {metric}')
    plt.plot(range(1,len(history.history[f'val_{metric}']) + 1),history.history[f'val_{metric}'], color='green', label=f'Validation {metric}')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.grid(True, which='both', linestyle='-')
    val_metric = history.history[f'val_{metric}']
    if metric == "loss":
        best_epoch = val_metric.index(min(val_metric)) + 1
    else:
        best_epoch = val_metric.index(max(val_metric)) + 1

    plt.scatter(best_epoch, val_metric[best_epoch - 1], color='blue', s=20, label=f'best epoch= {best_epoch}')
    plt.legend(loc='upper right', fontsize='x-small')
    plt.show()
    return



def plot_roc_curve(model, X, y, class_names):
    """
    Plots the ROC curve for the validation set without the macro-average.

    Parameters:
    - model: Trained model used to predict the validation set.
    - X:  feature data.
    - y: True labels for the dataset set.
    - class_names: List of class names for labeling the ROC curve plot.
    """

    # Get model predictions for the validation set
    y_pred = model.predict(X)
    y_true = np.argmax(y, axis=1)

    # Binarize the output for multi-class ROC curve
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    y_pred_bin = y_pred

    # Compute ROC curve and ROC AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {} (area = {:.2f})'.format(class_name, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    plt.show()

    return roc_auc



def plot_confusion_matrix(model, X, y, class_names, normalize=False):
    """
    Generates and plots the confusion matrix.

    Parameters:
    - model: Trained model used to predict the dataset.
    - X: Feature data (validation or test set).
    - y: True labels for the dataset.
    - class_names: List of class names for labeling the confusion matrix.
    - normalize: If True, normalize the confusion matrix by dividing by the sum of rows.
    """
    # Get model predictions
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Normalize confusion matrix if specified
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

    # Customize the plot
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
