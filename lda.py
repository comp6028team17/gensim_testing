import logging, gensim, bz2
import json
from gensim import corpora, models, similarities

#load text corpus from json doc
jsonDoc = 'sports.arts.computers.buisness.management.home.society.science.json'

def loadJSONcorpus(filename):
    doc_words = []
    #load JSON file
    with open(filename) as jFile:
        jData = json.load(jFile)
        #for each document in JSON doc arr
        for document in jData:
            #for each item in words
            #for key, value document['words'].interitems():
            doc_words.append(document['words'])
    return doc_words

def createDictionary(corpus_texts):
    dictionary = corpora.Dictionary(corpus_texts)
    dictionary.save(jsonDoc + '.dict')
    return dictionary

def loadDictionary(filename):
    return gensim.corpora.Dictionary.load_from_text(filename + '.dict')

def createCorpus(filename):
    corpus_texts = loadJSONcorpus(filename)
    dictionary = createDictionary(corpus_texts)
    corpus = [dictionary.doc2bow(text) for text in corpus_texts]
    #save corpus
    corpora.MmCorpus.serialize(jsonDoc + '.mm', corpus)
    return corpus, dictionary

def loadCorpus(filename):
    return gensim.corpora.MmCorpus(filename + '.mm'), gensim.corpora.Dictionary.load(filename + '.dict')




#Set log conf
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

mm, id2word = loadCorpus(jsonDoc)
# mm, id2word = createCorpus(jsonDoc)


##NEED TO FILTER STOP WORDS
# tfidf = models.TfidfModel(mm)

print(mm)

#gen model
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)

lda.print_topics(20)
