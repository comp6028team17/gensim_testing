import json
import string
from itertools import combinations_with_replacement
import inspect, os
from bs4 import BeautifulSoup
import numpy as np
import sklearn

with open(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/stop_words.json") as f:
	stoplist = set(json.load(f))

def make_proportional(X, y, min_size = 100):
    counts = np.bincount(y)
    min_size = min(counts[counts>min_size])
    class_inds = (np.where(y == i)[0] for i in range(max(y)+1))
    indices = np.array(
        [np.random.choice(inds, size=min_size, replace=False, p=None) for inds in class_inds 
         if len(inds) >= min_size]
    ).flatten()
    return (X[indices], y[indices])

def loadSplitJsonLines(fname):
	''' Load every file matching the sequence fname.aa, fname.ab ... fname.zz
		Parse each line in each file one by one as JSON, and yield it
	'''
	numyielded = 0
	for s in (''.join(x) for x in combinations_with_replacement(string.ascii_lowercase, 2)):
		try:
			loadfn = fname+s
			with open(loadfn, 'r') as f:
				numyielded += 1
				for l in f.readlines():
					yield json.loads(l)
			
		except IOError as e:
			if numyielded == 0:
				raise Exception("No files matching {}.aa".format(fname))
			return


def main():
	""" Print an example of one loaded file """
	for x in loadSplitJsonLines('../docs/sites.jl.'):
		print x
		break

if __name__ == '__main__':
	main()


def encode_dmoz_categories(dmoz_categories):
	# Build a list of all top-level topics
	topcategories = set(topic[0] for topic in dmoz_categories)
	# Represent the topics in an alternative way
	heirarchal_categories = lambda max_depth: [['; '.join(topics[:ti+1]) for ti, t in enumerate(topics) if ti < max_depth] for topics in dmoz_categories]
	# Top categories
	top_categories = [x[0] for x in heirarchal_categories(1)]
	# Encoder
	dmoz_encoder = sklearn.preprocessing.LabelEncoder().fit(top_categories)
	# Classes
	classes = dmoz_encoder.transform(top_categories)
	return (classes, top_categories, dmoz_encoder)
