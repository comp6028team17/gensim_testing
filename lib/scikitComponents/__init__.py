from sklearn.base import BaseEstimator, TransformerMixin
import gensim
import numpy as np
import itertools
import collections

class LDAModel(BaseEstimator, TransformerMixin):

	def __init__(self, dictionary, num_topics):
		self.dictionary = dictionary
		self.num_topics = num_topics

	def fit(self, corpus, y=None):
		""" Build an LDA Model using a gensim corpus """
		self._model = gensim.models.ldamodel.LdaModel(
			corpus=corpus, 
			id2word=self.dictionary,
			num_topics = self.num_topics,
			passes = 1,
			chunksize = 100)
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


class MetaMatrixBuilder(BaseEstimator, TransformerMixin):
	def __init__(self, top_min_members = 0):
		self.topic_min_members = 0

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		""" Convert a TransformedCorpus object to a matrix of corpus vs topic index """
		lists = [set(x.get('keyphrases', [])) | set(x.get('keywords', [])) for x in X]
		sanitize = lambda word: word.strip()
		counts = collections.Counter(sanitize(word) for word in itertools.chain(*lists))
		words = [x for x in counts if counts[x] > 4 and counts[x] < len(X)-4 and x != '']

		features = np.zeros((len(X), len(words)), dtype=int)
		for i, doc in enumerate(lists):
		    features[i,[words.index(x) for x in doc if x in words]] += 1

		# return features[:,np.sum(features > 0, axis=0)>self.topic_min_members]
		return features
