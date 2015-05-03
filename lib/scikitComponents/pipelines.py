import sklearn
import sklearn.neighbors
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import sklearn.grid_search
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.preprocessing

from . import *

class FullClassifier(BaseEstimator, ClassifierMixin):
    def __init__(dictionary, corp_type, feature, classifier, lda_num_topics):
        pass


def make_classifier(dictionary, corp_type='both', feature='tfidf', classifier = 'svc', lda_num_topics = 20, class_weight='auto'):
    clfnames = {
    'svc': 'SVM',
    'svc_gridsearch': 'SVM',
    'trees': 'D.Trees',
    'nb': 'N.Bayes',
    'nb_bin': 'BernoulliBayes',
    'sgd': 'SGD'
    }

    corps = {
        'body': 0,
        'meta': 1,
        'both': 2
    }

    corpnames = ['body', 'meta', 'both']

    if classifier == 'svc_gridsearch': 
        param_grid = [{'C': [0.05, 0.08, 0.1, 0.11, 0.13]}]
        #param_grid = [{'C': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]}]
        svc = sklearn.svm.LinearSVC(class_weight=class_weight)
        clf = sklearn.grid_search.GridSearchCV(svc, param_grid, verbose=3)
    elif classifier == 'svc':
        clf = sklearn.svm.LinearSVC(class_weight=class_weight, C = 0.08)
    elif classifier == 'knn': 
        clf = sklearn.neighbors.KNeighborsClassifier()
    elif classifier == 'trees':
        clf = sklearn.ensemble.ExtraTreesClassifier(random_state=0, n_estimators=100, oob_score=True, bootstrap=True, n_jobs=4)
    elif  classifier == 'nb':
        clf = sklearn.naive_bayes.MultinomialNB()
        #clf = sklearn.naive_bayes.GaussianNB()
    elif classifier == 'nb_bin':
        clf = sklearn.naive_bayes.BernoulliNB(binarize=0.0)
    elif classifier == 'sgd':
        clf = sklearn.linear_model.SGDClassifier(loss='log', class_weight=class_weight)


    if feature == 'tfidf':
        extract = Pipeline([
        ('tfidf', TFIDFModel(dictionary)),
        ('matrix', TopicMatrixBuilder(len(dictionary.items()), keep_all=True))
        ])
    elif feature == 'lda':
        extract = Pipeline([
        ('lda_model', LDAModel(dictionary, lda_num_topics)),
        ('matrix_builder', TopicMatrixBuilder(lda_num_topics))
        ])
    elif feature == 'count':
        extract = Pipeline([
        ('matrix_builder', CorpusCount(len(dictionary)))
        ])

    pipe = Pipeline([
        ('selection', ItemPicker(corps[corp_type])),
        ('feature', extract),
        #('scale', sklearn.preprocessing.StandardScaler()),
        ('classifier', clf)
    ])

    pipe.name = "{}({}) - {}".format(feature, corp_type, clfnames[classifier])

    return pipe



