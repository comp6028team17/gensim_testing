import logging, gensim, bz2

corporaDictFname = 'corpora.dictionary.txt'
tfidfFname = 'tfidf.mm'

#Set log conf
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load word mappings
id2word = gensim.corpora.Dictionary.load_from_text(corporaDictFname)

#load corpus iterator
mm = gensim.corpora.MmCorpus(tfidfFname)

print(mm)

#gen model
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)

lda.print_topics(20)
