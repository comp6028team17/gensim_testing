from sklearn.base import BaseEstimator, TransformerMixin
import gensim
import numpy as np
import itertools
import collections
import os
import inspect
import json

# Load the stopwrod list. Should probably find a nicer way of doign this!
# (import datastuff module, or something)
with open(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/../stop_words.json") as f:
    stoplist = set(json.load(f))


class LDAModel(BaseEstimator, TransformerMixin):

    def __init__(self, dictionary, num_topics):
        self.dictionary = dictionary
        self.num_topics = num_topics

    def fit(self, corpus, y=None):
        """ Build an LDA Model using a gensim corpus """
        self._model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=1,
            chunksize=100)
        return self

    def transform(self, corpus, y=None):
        """ Return a gensim TransformedCorpus object """
        return self.model[corpus]

    @property
    def model(self):
        return self._model


class ItemPicker(BaseEstimator, TransformerMixin):

    def __init__(self, index):
        self.index = index

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [x[self.index] for x in X]


class TopicMatrixBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, num_topics, topic_min_members=0):
        self.num_topics = num_topics
        self.topic_min_members = 0

    def fit(self, X, y=None):
        return self

    def transform(self, transformed_corpus, y=None):
        """ Convert a TransformedCorpus object to a matrix of corpus vs topic index """
        features = np.zeros((len(transformed_corpus), self.num_topics))
        totals = np.zeros(self.num_topics)
        for i, doc in enumerate(transformed_corpus):
            for topic, p in doc:
                features[i, topic] = p
                totals[topic] += p

        return features[:, np.sum(features > 0, axis=0) > self.topic_min_members]


class MetaMatrixBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, use_keywords=True, use_keyphrases=True, use_description=False, min_word_count=4, stem=False):
        self.use_keywords = use_keywords
        self.use_keyphrases = use_keyphrases
        self.use_description = use_description
        self.min_word_count = min_word_count
        self.featureset = []
        self.stem = stem
        if stem:
            import nltk.stem.porter
            stemmer = nltk.stem.porter.PorterStemmer()
            self.sanitize = lambda word: stemmer.stem(word.strip())
        else:
            self.sanitize = lambda word: word.strip()

    def extract_meta(self, X):
        """ Extract the required bits of metadata from each document objet in X 
            returns a generator of generators of words
        """
        for meta in X:
            lists = []
            if self.use_keywords:
                lists.append(meta.get('keywords', []))
            if self.use_keyphrases:
                lists.append(meta.get('keyphrases', []))
            if self.use_description:
                lists.append(meta.get('description', []))

            yield (self.sanitize(word) for word in itertools.chain(*lists))

    def fit(self, X, y=None):
        """ Given a list of metadata objects, generate maps between tokens and IDs """
        lists = self.extract_meta(X)
        counts = collections.Counter(itertools.chain(*lists))
        self.featureset = {
            w: i for i, w in
            enumerate(set(x for x in counts 
                          if counts[x] > self.min_word_count and x not in stoplist and x != ""))
        }
        self.inverse_featureset = {
            i: w for i, w in
            enumerate(set(x for x in counts 
                          if counts[x] > self.min_word_count and x not in stoplist and x != ""))
        }

        return self

    def inverse_transform(self, X, y=None):
        """ Given the output of transform(), regenerate the list of words"""
        invert = lambda row: ([self.inverse_featureset[i]] * count for i, count in enumerate(row) if count > 0)
        return [list(itertools.chain.from_iterable(invert(row))) for row in X]

    def transform(self, X, y=None):
        """ Convert a TransformedCorpus object to a matrix of corpus vs topic index """
        features = np.zeros((len(X), len(self.featureset)), dtype=int)
        allmeta = self.extract_meta(X)
        for i, meta in enumerate(allmeta):
            tokens = (x for x in (self.featureset.get(word, None)
                                  for word in meta) if x is not None)
            features[i, :] += np.bincount(list(tokens), minlength=len(self.featureset))

        return features
