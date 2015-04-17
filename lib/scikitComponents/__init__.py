from sklearn.base import BaseEstimator, TransformerMixin
import gensim
import numpy as np

class LDAModel(BaseEstimator, TransformerMixin):

	def __init__(self, dictionary, num_topics):
		self.dictionary = dictionary
		self.num_topics = num_topics

	def fit(self, corpus, y=None):
		""" Build an LDA Model using a gensim corpus """
		self._model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=self.dictionary, num_topics = self.num_topics, passes = 1, chunksize = 100)
		return self

	def transform(self, corpus, y=None):
		""" Return a gensim TransformedCorpus object """
		return self.model[corpus]

	@property
	def model(self):
		return self._model


class TopicMatrixBuilder(BaseEstimator, TransformerMixin):
	def __init__(self, num_topics, topic_min_members = 0):
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

		return features[:,np.sum(features > 0, axis=0)>self.topic_min_members]

