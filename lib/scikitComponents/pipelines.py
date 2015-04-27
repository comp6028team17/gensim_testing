import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
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



def make_classifier(dictionary, classifier = 'svc', body_kind = 'tfidf', meta_kind='tfidf', meta_selection_flags = 13, lda_num = 30):
    if classifier == 'svc': 
        classifier = sklearn.svm.LinearSVC()
    elif classifier == 'svcproba': 
        classifier = sklearn.svm.SVC(kernel='linear', probability=True, max_iter=1000)
    elif classifier == 'trees':
        classifier = sklearn.ensemble.ExtraTreesClassifier(random_state=0, n_estimators=100, oob_score=True, bootstrap=True, n_jobs=4)

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
        
    return Pipeline([
        ('union', FeatureUnion(features)), 
        ('clf', classifier)])



