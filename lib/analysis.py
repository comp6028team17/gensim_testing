import matplotlib.pyplot as plt
import numpy as np
import collections
import sklearn
from sklearn.pipeline import Pipeline
import sklearn.metrics

def red(s):
    return "\x1b[31m{}\x1b[0m".format(s)

class ExampleInspector():
    def __init__(self, dictionary, clf, X_test, y_test, predicted, meta, ind_test, dmoz_encoder):
        self.dictionary = dictionary
        self.y_test = y_test
        self.predicted = predicted
        self.dmoz_encoder = dmoz_encoder
        self.ind_test = ind_test
        self.meta = meta
        self.clf = clf
        self.X_test = X_test

    def get_confusion(self, actually_was=None, predicted_as=None, predicted_on_guess = 1, only_failures=False):
        
        if only_failures:
            filtered = np.where(1-np.equal(self.predicted, self.y_test))[0]
        else:
            filtered = np.arange(len(self.y_test))
        
        fail_name = as_name = was_name = guess_name = ""
        filta = lambda c: True
        filtb = lambda c: True
        
        if only_failures:
            fail_name = "incorrectly classified "
        if predicted_as is not None:
            as_name = "predicted as '{}'".format(self.dmoz_encoder.inverse_transform(predicted_as))        
            filta = (lambda c: c == predicted_as)
        if actually_was is not None:
            was_name = "of class '{}'".format(self.dmoz_encoder.inverse_transform(actually_was))
            filtb = (lambda c: c == actually_was)
        if predicted_on_guess > 1:
            guess_name = "on guess {}".format(predicted_on_guess)
        
        name = " ".join([text for text in [fail_name, "samples", was_name, as_name, guess_name] if text])
        print name+"\n"
        
        meta_test = [self.meta[i] for i in self.ind_test]
        guess_nums = np.fliplr(np.argsort(self.clf.decision_function(self.X_test)))
        guesses = self.dmoz_encoder.inverse_transform(guess_nums)
        probs = self.clf.decision_function(self.X_test)
        transformed = Pipeline(self.clf.steps[:2]).transform(self.X_test)
        weights = self.clf.steps[2][1].coef_
        
        
        
        for x in [x for x in filtered if filta(guess_nums[x][predicted_on_guess-1]) and filtb(self.y_test[x])]:
            redright = lambda gs: [(red(g) if g == meta_test[x][1][0] else g) for g in gs]
            topg = guesses[x][:3]
            print meta_test[x][0], ", ".join((meta_test[x][1][:3]))
            print "Predicted as:", ", ".join(redright(topg))
            
            trf = transformed[x] * weights[self.predicted[x]]
            inds = np.argsort(trf)[::-1]
            best = [self.dictionary[i] for i in inds][:9]
            print "{} words: {}".format(topg[0].capitalize(), ", ".join(best))
            print

def plot_confusion_matrix(cm, title='Confusion matrix', labels=None, cmap=plt.cm.Blues, showwarnings=True, scores = None):
    """ Plot a confusion matrix as a matrix of coloured squares """

    plt.figure(figsize=(7, 6))
    im = cm.astype(float)
    #print np.mean(im, axis=1)
    im = (im.T/np.sum(im, axis=1)).T * 100
    #print im
    #im[np.where(cm!=0)] = np.log(im[cm!=0])
    im[np.where(im!=0)] = np.log(im[im!=0])
    #im[np.where(im==0)] = 0
    labels = labels if labels is not None else range(len(cm))
    plt.imshow(im, interpolation='nearest', cmap=cmap)
    if title: plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for y in range(cm.shape[1]):
        rowsum = float(np.sum(cm[y, :]))
        if rowsum > 0:
            if scores is not None:
                txt = 'F1 score: {:.2}'.format(scores[y])
                plt.text(cm.shape[0]+0.3, y, txt, va='center')
        for x in range(cm.shape[0]):
            if im[y, x] > (np.max(im)*0.5):
                col = 'w'
            else:
                col = 'k'
            if showwarnings == True and x != y and cm[y, x] > cm[y, y]/5:
                col = 'r'
            plt.text(x, y, cm[y, x], color = col, ha='center', va='center', weight='bold')


def plot_proportion_investigation(predicted, dmoz_encoder, y_test, adjust_labels=False):
    """ Look at the proportion of items in a test set, versus their scores """
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

def accuracy_of_top_n_guesses(clf, X, y, n=None):
    """ Given a scikit estimator, input data and output classes,
    return how well the algorithm finds the correct classes in its
    top 1, 2, 3 guesses ... top N guesses."""

    n = n or max(y)
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X)
        labels = np.argsort(probs, axis=1)[:, -n:][:, ::-1]
    else:
        labels = np.fliplr(np.argsort(clf.decision_function(X)))
    
    results = np.array([label == row for label, row in zip(y, labels)])

    scores = [
        np.sum(np.max(results[:, 0:i], axis=1)) / float(len(results))
        for i in range(1, n+1)
    ]
    return scores


def print_score_accuracy(scores):
    print "Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)