#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
import gensim
import numpy as np
import itertools
import collections
import os
import inspect
import json
import re
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


class CorpusCount(BaseEstimator, TransformerMixin):
    def __init__(self, num_words):
        self.num_words = num_words
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        ret = np.zeros((len(X), self.num_words))
        for i, row in enumerate(X):
            ret[i] = np.bincount([x[0] for x in row], minlength = self.num_words)
        return ret


class TopicMatrixBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, num_topics, topic_min_members=0, keep_all = False):
        self.num_topics = num_topics
        self.topic_min_members = 0
        self.keep_all = keep_all
    def fit(self, X, y=None):
        return self

    def transform(self, transformed_corpus, y=None):
        """ Convert a TransformedCorpus object to a matrix of corpus vs topic index """
        features = np.zeros((len(transformed_corpus), self.num_topics))
        for i, doc in enumerate(transformed_corpus):
            for topic, p in doc:
                features[i, topic] = p
        if self.keep_all:
            return features
        else:
            return features[:, np.sum(features > 0, axis=0) > self.topic_min_members]


class TFIDFModel(BaseEstimator, TransformerMixin):
    def __init__(self, id2word):
        self.id2word = None
        self.tfidf = None
    def fit(self, X, y=None):
        self.tfidf = gensim.models.tfidfmodel.TfidfModel(X, id2word=self.id2word)
        return self
    def transform(self, X, y=None):
        return self.tfidf[X]



class MetaSanitiser(BaseEstimator, TransformerMixin):
    def __init__(self, stem = False, meta_selection_flags=7):
        self.stem = stem
        self.meta_selection_flags = meta_selection_flags

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        """ Extract the required bits of metadata from each document objet in X 
            returns a generator of generators of words
        """
        # Set up a sanitizer
        if self.stem:
            # We can stem words, i.e. remove suffixies.
            # this means that words like 'sport', 'sporting' and 'sports'
            # would all be treated as 'sport'
            import nltk.stem.porter
            stemmer = nltk.stem.porter.PorterStemmer()
            sanitize = lambda word: stemmer.stem(re.sub(r'\s+', r' ', word.strip()))
        else:
            sanitize = lambda word: re.sub(r'\s+', r' ', word.strip())
        out = []
        for meta in X:
            lists = []
            if self.meta_selection_flags & 1:
                lists.append(meta.get('keywords', []))
            if self.meta_selection_flags & 2:
                lists.append(meta.get('keyphrases', []))
            if self.meta_selection_flags & 4:
                lists.append(meta.get('description', []))
            if self.meta_selection_flags & 8:
                lists.append(meta.get('title', []))

            words = (sanitize(word) for word in list(itertools.chain.from_iterable(lists)))
            filtered = [word for word in words if word not in stoplist and word != ""]

            out.append(filtered)

        return out

class CountMatrixBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, min_word_count = 4):
        self.min_word_count = min_word_count
        self.featureset = []
           

    def fit(self, X, y=None):
        """ Given a list of metadata objects, generate maps between tokens and IDs """
        counts = collections.Counter(itertools.chain(*X))
        self.featureset = {
            w: i for i, w in
            enumerate(set(x for x in counts 
                          if counts[x] > self.min_word_count))
        }
        self.inverse_featureset = {
            i: w for i, w in
            enumerate(set(x for x in counts 
                          if counts[x] > self.min_word_count))
        }

        return self

    def inverse_transform(self, X, y=None):
        """ Given the output of transform(), regenerate the list of words"""
        invert = lambda row: ([self.inverse_featureset[i]] * count for i, count in enumerate(row) if count > 0)
        return [list(itertools.chain.from_iterable(invert(row))) for row in X]

    def transform(self, X, y=None):
        """ Convert a TransformedCorpus object to a matrix of corpus vs topic index """
        features = np.zeros((len(X), len(self.featureset)), dtype=int)
        for i, meta in enumerate(X):
            tokens = (x for x in (self.featureset.get(word, None)
                                  for word in meta) if x is not None)
            for t in tokens:
                features[i, t] += 1

        return features
