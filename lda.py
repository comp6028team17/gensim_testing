import logging, gensim, bz2
import json
from gensim import corpora, models, similarities

#load text corpus from json doc
jsonDoc = 'topics.sport.linux.computers.art.history.language.society.json'

loadJSONcorups(filename):
    doc_words = []
    #load JSON file
    with open(filename) as jFile:
        jData = json.load(jile)
        #for each document in JSON doc arr
        for document in jData:
            #for each item in words
            #for key, value document['words'].interitems():
            doc_words.append(document['words'])

    return doc_words

createDictionary(corpus_texts):
    dictionary = corpora.Dictionary(corpus_texts)
    dictionary.save(filename + '.dict')
    return dictionary

loadDictionary(filename):
    return gensim.corpora.Dictionary.load_from_text(filename + '.dict')

createCorpus(filename):
    corpus_texts = loadJSONcorpus(filename)
    dictionary = createDictionary(corpus_texts)
    corpus = [dictionary.doc2bow(text) for text in corpus_texts]
    #save corpus
    corpora.MmCorpus.serialize(filename + '.mm', corpus)
    return corpus

loadCorpus(filename):
    return gensim.corpora.MmCorpus(filename + '.mm')


#Set log conf
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load word mappings
id2word = loadDictionary(jsonDoc)

#load corpus iterator
# mm = gensim.corpora.MmCorpus(tfidfFname)
mm = loadCorpus(jsonDoc)

print(mm)

#gen model
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)

lda.print_topics(20)
