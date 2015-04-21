import matplotlib.pyplot as plt
import numpy as np
import collections

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


def plot_proportion_investigation(predicted, dmoz_encoder, y_test, adjust_labels=False):
    category_scores = collections.defaultdict(float)
    category_counts = collections.defaultdict(float)

    for real, pred in sorted(zip(dmoz_encoder.inverse_transform(y_test), dmoz_encoder.inverse_transform(predicted))):
        category_scores[real] += 1 if real == pred else 0
        category_counts[real] += 1 
    data = np.array(sorted([
                (category_counts[cat]/len(y_test), category_scores[cat] / category_counts[cat]) 
                for cat in category_counts.keys()
            ])).T
    plt.plot(*data, marker='o')
    plt.xlabel('proportions')
    plt.ylabel('scores')
    plt.title('Proportion of categories in test-set versus category score')
    i = -1
    for s, tx, ty in zip(category_counts.keys(), *data):
        if adjust_labels:
            tx = -0.0005+tx+i*0.001
        else:
            tx+=0.001
        plt.text(tx, ty, s)
        i*=-1
    plt.show()