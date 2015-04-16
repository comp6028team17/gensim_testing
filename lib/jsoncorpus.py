
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

class TagStripper(HTMLParser):
    """ Class which strips data from HTML tags """
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

    @classmethod
    def strip_tags(cls, html):
        s = TagStripper()
        s.feed(html)
        return s.get_data()

SiteWords = namedtuple('SiteWords', ['body', 'meta'])
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
    for item in soup.findAll(text=lambda text:isinstance(text, Comment)):
        item.extract()

    # Define a function to strip all punctuation from a string
    strip_punctuation = lambda x: re.sub(r'[^A-Za-z0-9- ]|(\b[\d-]+\b)', '', x)

    # Define a function to check of if a tag is a meta tag we care about
    is_meta_tag = lambda tag: tag.name == 'meta' and tag.attrs.get('name', None) in ['description', 'keywords','author']

    # Strip all tags from the body, strip all punctuation from the text
    # Then split in to a list of words
    body_content = strip_punctuation(TagStripper.strip_tags(str(soup.body))).split()


    # Build a dictionary from the metatags metadata, linking name->content
    meta_tags = {
        tag.attrs['name']: strip_punctuation(tag.attrs.get('content', '')).split()
        for tag in soup.findAll(is_meta_tag)
    }

    # Add the title to the metadata
    meta_tags['title'] = strip_punctuation(soup.head.title.text).split() if soup.head and soup.head.title else ''

    return SiteWords(body_content, meta_tags)



def get_processed_data(fnames, max_i=None):
    """ Process all the files, ignoring files with unicode problems... """

    all_html = (d['html'] for d in datastuff.loadSplitJsonLines(fnames+"."))
    for i, html in enumerate(all_html):
        if i%20 == 0: print i
        if i==max_i: break
        try: yield process_html(html)
        except UnicodeDecodeError: continue

def load_or_create(fnames):
    data = None
    try:
        # Load a saved dictionary if it exists
        dictionary = corpora.Dictionary.load('dictionary.dict')
    except IOError:
        print "Building new dictionary"
        if not data:
            data = list(get_processed_data(fnames))

        # Otherwise, build a new gensim dictionary
        dictionary = corpora.Dictionary((
                (word for word in itertools.chain(words.body, *words.meta.values())
                    if word not in datastuff.stoplist)
                for words in data))
        dictionary.filter_extremes()
        dictionary.compactify()
        dictionary.save('dictionary.dict')

    try:
        # Load the Matrix Market corpus files if they exist
        body_corpus = corpora.mmcorpus.MmCorpus('body_corpus.mm')
        meta_corpus = corpora.mmcorpus.MmCorpus('meta_corpus.mm')
    except IOError:
        print "Building new corpora"
        if not data:
            data = list(get_processed_data(fnames))
        # Otherwise, construct the corpora using the dictionary
        body_data = (dictionary.doc2bow(doc) for doc in ((word for word in words.body if word not in datastuff.stoplist) for words in data))
        body_corpus = corpora.mmcorpus.MmCorpus.serialize('test_body.mm', body_data, id2word=dictionary)
        meta_data = (dictionary.doc2bow(doc) for doc in ((word for word in itertools.chain(*words.meta.values()) if word not in datastuff.stoplist) for words in data))
        meta_corpus = corpora.mmcorpus.MmCorpus.serialize('test_meta.mm', meta_data, id2word=dictionary)

    return (dictionary, body_corpus, meta_corpus)

if __name__ == '__main__':
    load_or_create("../docs/sites.jl")