
from gensim import corpora, models, similarities
import datastuff
from collections import Counter
import os
import re
import traceback

class JsonLinesCorpus(corpora.textcorpus.TextCorpus):
    def __init__(self, name, fname):
        self.fname = fname
        self.name = name
        super(JsonLinesCorpus, self).__init__("") # not super sure why 


    def get_filterwords(self, name, fname):
        dictfname = '{}.full.dict'.format(name)

        try:
            dictionary = corpora.Dictionary.load(dictfname)
        except IOError:
            print "Building filter list..."
            dictionary = corpora.Dictionary(d['words'] for d in datastuff.loadSplitJsonLines(fname+"."))
            dictionary.compactify()
            dictionary.save(dictfname)

        filterwords = set((dictionary[x] for x in (tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1))).union(datastuff.stoplist)
        
        return filterwords


    def get_texts(self):
        self.length = 0
        filterwords = self.get_filterwords(self.name, self.fname)

        def wordok(w):
            return w not in filterwords and len(w) > 1 and re.match('^[\d-]+$', w) == None

        print "Building corpora..."
        for j in datastuff.loadSplitJsonLines(self.fname+"."):

            words = [w for w in j['words'] if wordok(w)]
            yield words
            self.length += 1

    def __len__(self):
        return self.length

def load_or_create(name, fname):
    ''' Build an MM corpus and dictionary out of a directory full of json-lines files '''
    mmname = "{}.mm".format(name)
    dname = "{}.dict".format(name)
    try:
        corpus = corpora.mmcorpus.MmCorpus(mmname)
        dictionary = corpora.Dictionary.load(dname)
        print "Succesfully loaded {} and {}".format(mmname, dname)
    except IOError:
        corp = JsonLinesCorpus(name, fname)
        corpora.mmcorpus.MmCorpus.serialize(mmname, corp)
        corp.dictionary.save(dname)
        corpus = corpora.mmcorpus.MmCorpus(mmname)
        dictionary = corpora.Dictionary.load(dname)
        print "Succesfully created {} and {}".format(mmname, dname)

    return (corpus, dictionary)


def main():
    """ Test! """
    corpus, dictionary = load_or_create("../corpus_test", "../docs/sites.jl")
    print corpus[0]


if __name__ == '__main__':
    main()
