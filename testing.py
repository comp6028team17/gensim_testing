from lib import jsoncorpus
import gensim

def main():
	corpus, dictionary = jsoncorpus.load_or_create('corpus', 'docs/sites.jl')


	model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, update_every=1)

	model


if __name__ == '__main__':
	main()



