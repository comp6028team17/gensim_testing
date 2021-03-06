
from gensim import corpora, models, similarities
import datastuff
from collections import Counter
import os
import re
import traceback
from HTMLParser import HTMLParser
from bs4 import BeautifulSoup, Comment
from collections import namedtuple
import itertools
import json


class TagStripper(HTMLParser):

    """ Class which strips data from HTML tags """

    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ' '.join(self.fed)

    @classmethod
    def strip_tags(cls, html):
        s = TagStripper()
        s.feed(html)
        return s.get_data()

ProcessedPage = namedtuple(
    'ProcessedPage', ['body', 'meta', 'url', 'topics', 'index'])


def process_html(html):
    """ Given a HTML page, return an object containing... {
        'body': a list of visible words in the body,
        'meta': a dictionary of lists of the words in the content of the description, author and keywords tags
    }"""

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html.lower())

    # Remove all <script> and <style tags> and <!-- comments -->
    for item in soup.findAll('script'):
        item.extract()
    for item in soup.findAll('style'):
        item.extract()
    for item in soup.findAll(text=lambda text: isinstance(text, Comment)):
        item.extract()

    # Define a function to strip all punctuation from a string
    def strip_punctuation(word):
        word = re.sub('[,;./\\|?()]', ' ', word.strip())
        word = re.sub(r'([^A-Za-z0-9- ]|(^\b[\d-]+\b$))', '', word)
        return word

    # Define a function to check of if a tag is a meta tag we care about
    is_meta_tag = lambda tag: tag.name == 'meta' and tag.attrs.get(
        'name', None) in ['description', 'keywords', 'author']
    is_keyword_tag = lambda tag: tag.name == 'meta' and tag.attrs.get(
        'name', None) in ['keywords']

    # Strip all tags from the body, strip all punctuation from the text
    # Then split in to a list of words
    body_content = strip_punctuation(
        TagStripper.strip_tags(str(soup.body))).split()

    # Build a dictionary from the metatags metadata, linking name->content
    meta_tags = {
        tag.attrs['name']: strip_punctuation(tag.attrs.get('content', '')).split()
        for tag in soup.findAll(is_meta_tag)
    }

    meta_tags.update({
        # Split list of keywords by commas, after removing any whitespace
        # around commas
        "keyphrases": re.sub(r',\s([\w\s]*)', r',\1', tag.attrs.get('content', '')).split(',')
        for tag in soup.findAll(is_keyword_tag)
    })

    # Add the title to the metadata
    meta_tags['title'] = strip_punctuation(
        soup.head.title.text).split() if soup.head and soup.head.title else ''

    return (body_content, meta_tags)


def get_processed_data(fnames, max_i=None, print_i=False):
    """ Process all the files, ignoring files with unicode problems... """

    for i, doc in enumerate(datastuff.loadSplitJsonLines(fnames + ".")):
        if i == 562:
            continue
        if print_i and i % 20 == 0:
            print i
        if i == max_i:
            break
        try:
            body, meta = process_html(doc['html'])
            yield ProcessedPage(body, meta, doc['url'], doc['topics'], i)
        except UnicodeDecodeError:
            continue


def stem_words(iterable):
    import nltk.stem.porter
    stemmer = nltk.stem.porter.PorterStemmer().stem
    for word in iterable:
        yield stemmer(word)


def load_or_create(fnames, prefix="", stemmed=False):
    """ Return the dictionary, body corpus and meta corpus, creating them if they don't exist """
    data = None
    dict_name = '{}dictionary.dict'.format(prefix)
    body_corpus_name = '{}body_corpus.mm'.format(prefix)
    meta_corpus_name = '{}meta_corpus.mm'.format(prefix)
    meta_file_name = '{}meta.json'.format(prefix)
    def loaddata(data):
        if not data:
            return list(get_processed_data(fnames, print_i=True))
        return data

    if stemmed:
        word_modifier = stem_words
    else:
        word_modifier = lambda w: w
    try:
        # Load a saved dictionary if it exists
        dictionary = corpora.Dictionary.load(dict_name)
    except IOError:
        print "Building new dictionary"
        data = loaddata(data)
        # Otherwise, build a new gensim dictionary
        dictionary = corpora.Dictionary((
            word_modifier(word for word in itertools.chain(words.body, *words.meta.values())
                          if word not in datastuff.stoplist)
            for words in data))
        dictionary.filter_extremes()
        dictionary.compactify()
        dictionary.save(dict_name)

    try:
        # Load the Matrix Market corpus files if they exist
        body_corpus = corpora.mmcorpus.MmCorpus(body_corpus_name)
        meta_corpus = corpora.mmcorpus.MmCorpus(meta_corpus_name)
    except IOError:
        print "Building new corpora"
        data = loaddata(data)
        # Otherwise, construct the corpora using the dictionary
        body_data = (dictionary.doc2bow(doc) for doc in (word_modifier(
            word for word in words.body if word not in datastuff.stoplist) for words in data))
        body_corpus = corpora.mmcorpus.MmCorpus.serialize(
            body_corpus_name, body_data, id2word=dictionary)
        meta_data = (dictionary.doc2bow(doc) for doc in (word_modifier(word for word in itertools.chain(
            *words.meta.values()) if word not in datastuff.stoplist) for words in data))
        meta_corpus = corpora.mmcorpus.MmCorpus.serialize(
            meta_corpus_name, meta_data, id2word=dictionary)

    try:
        with open(meta_file_name) as f:
            dmoz_data = json.load(f)
    except IOError:
        print "Building meta data file"
        data = loaddata(data)
        dmoz_data = {
            'urls': [d.url for d in data],
            'dmoz_categories': [d.topics for d in data],
            'meta': [d.meta for d in data]
        }
        with open(meta_file_name, 'w') as f:
            json.dump(dmoz_data, f)

    print "Done."
    return (dictionary, body_corpus, meta_corpus, dmoz_data)

if __name__ == '__main__':
    pass
    #for x in get_processed_data("../docs/sites.jl", print_i=dTrue, max_i=20):
    #    print x

    # load_or_create("../docs/sites.jl")
