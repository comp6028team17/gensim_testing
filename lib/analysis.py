import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, title='Confusion matrix', labels=None, cmap=plt.cm.Blues):
    labels = labels if labels is not None else range(len(cm))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')