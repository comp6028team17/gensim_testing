import json
import string
from itertools import combinations_with_replacement
import inspect, os
from bs4 import BeautifulSoup
import numpy as np
import sklearn

with open(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/stop_words.json") as f:
    stoplist = set(json.load(f))

def make_proportional(y, min_size = 100):
    counts = np.bincount(y)
    min_size = min(counts[counts>min_size])
    class_inds = (np.where(y == i)[0] for i in range(max(y)+1))
    indices = np.array(
        [np.random.choice(inds, size=min_size, replace=False, p=None) for inds in class_inds 
         if len(inds) >= min_size]
    ).flatten()
    return (indices, y[indices])

def tt_split(X, y, proportional=False, min_size = 1, test_size = 0.25):
    if proportional:
        inds, prop_y = make_proportional(np.array(X), y, min_size=min_size)
        prop_X = X[inds]
        ind_train, ind_test = next(iter(sklearn.cross_validation.StratifiedK(prop_y, test_size = test_size)))
        X_train , X_test, y_train, y_test = (prop_X[ind_train], prop_X[ind_test], prop_y[ind_train], prop_y[ind_test])
    else:
        X_train, X_test, y_train, y_test, ind_train, ind_test = sklearn.cross_validation.train_test_split(X, y, range(len(X)))

    return (X_train, X_test, y_train, y_test, ind_train, ind_test)

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


def encode_dmoz_categories(dmoz_categories, corpus, level=0, minwords = 0):
    dmoz_categories = np.array(dmoz_categories)

    # Replace category of empty websites with 'none'
    if minwords > 0:
        empties = [i for i, doc in enumerate(corpus) if len(doc) < minwords]
        if len(empties) > 0:
            dmoz_categories[empties] = [['none']]

    # Build a list of all top-level topics
    topcategories = set(topic[min(level, len(topic)-1)] for topic in dmoz_categories)
    # Represent the topics in an alternative way
    heirarchal_categories = lambda max_depth: [['; '.join(topics[:ti+1]) for ti, t in enumerate(topics) if ti < max_depth] for topics in dmoz_categories]
    # Top categories
    #top_categories = [x[min(level, len(x)-1)] for x in heirarchal_categories(2)]
    top_categories = [x[min(level, len(x) - 1)] for x in dmoz_categories] #[x[min(level, len(x)-1)] for x in heirarchal_categories(1)]
    # Encoder
    dmoz_encoder = sklearn.preprocessing.LabelEncoder().fit(top_categories)
    # Classes
    classes = dmoz_encoder.transform(top_categories)



    return (classes, top_categories, dmoz_encoder)
