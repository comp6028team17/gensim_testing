import sklearn
import sklearn.neighbors
from sklearn.pipeline import Pipeline, FeatureUnion
import sklearn.grid_search
import sklearn.naive_bayes

from . import *
def make_lda_pipeline(dictionary, num_topics):
    return Pipeline([
        ('lda_model', LDAModel(dictionary, num_topics)),
        ('matrix_builder', TopicMatrixBuilder(num_topics))
        ])

def make_meta_count_pipeline(meta_selection_flags):
    return Pipeline([
        ('meta_sanitiser', MetaSanitiser(meta_selection_flags = meta_selection_flags)),
        ('matrix_builder', MetaMatrixBuilder())
        ])

def make_tfidf_matrix_pipeline(dictionary):
    return Pipeline([
        ('tfidf', TFIDFModel(dictionary)),
        ('matrix', TopicMatrixBuilder(len(dictionary.items()), keep_all=True))
        ])



def make_classifier(dictionary, classifier = 'svc', body_kind = 'tfidf', meta_kind='tfidf', meta_selection_flags = 13, lda_num = 20):
    clfnames = {
    'svc': 'SVM',
    'svc_gridsearch': 'SVM',
    'trees': 'DecisionTrees',
    'nb': 'NaiveBayes'
    }
    bodynames = {
    'tfidf': "TfIdf(body)",
    'lda': 'LDA(body)'
    }
    metanames = {
    'tfidf': "TfIdf(meta)",
    'count': "Count(meta)"
    }


    if classifier == 'svc_gridsearch': 
        param_grid = [{'C': [0.1, 0.16, 0.2, 0.3, 0.4, 0.5, 1]}]
        #param_grid = [{'C': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]}]
        svc = sklearn.svm.LinearSVC(class_weight='auto')
        clf = sklearn.grid_search.GridSearchCV(svc, param_grid, verbose=3)
    elif classifier == 'svc':
        clf = sklearn.svm.LinearSVC(class_weight='auto', C = 0.16)
    elif classifier == 'knn': 
        clf = sklearn.neighbors.KNeighborsClassifier()
    elif classifier == 'trees':
        clf = sklearn.ensemble.ExtraTreesClassifier(random_state=0, n_estimators=100, oob_score=True, bootstrap=True, n_jobs=4)
    elif  classifier == 'nb':
        clf = sklearn.naive_bayes.MultinomialNB()

    body_pipe = meta_pipe = None
    
    features = []
    
    if body_kind == 'tfidf':
        body_pipe = make_tfidf_matrix_pipeline(dictionary)
    elif body_kind == 'lda':
        body_pipe = make_lda_pipeline(dictionary, lda_num)
    if meta_kind == 'tfidf':
        meta_pipe = make_tfidf_matrix_pipeline(dictionary)
        meta_i = 1
    elif meta_kind == 'count':
        meta_pipe = make_meta_count_pipeline(meta_selection_flags)
        meta_i = 2

    if body_pipe:
        features.append(('body', Pipeline(
                [('selector', ItemPicker(0)), ('feature', body_pipe)])))
    if meta_pipe:
        features.append(('meta', Pipeline(
                [('selector', ItemPicker(meta_i)), ('feature', meta_pipe)])))
    assert len(features) > 0

    pipe = Pipeline([
        ('union', FeatureUnion(features)), 
        ('clf', clf)])

    pipe.name = " ".join(filter(lambda x: x, (bodynames.get(body_kind, None), metanames.get(meta_kind, None)))) + " " + clfnames[classifier]


    return pipe



