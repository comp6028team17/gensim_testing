import json
import string
from itertools import combinations_with_replacement
import inspect, os


with open(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/stop_words.json") as f:
	stoplist = set(json.load(f))


def loadSplitJsonLines(fname):
	''' Load every file matching the sequence fname.aa, fname.ab ... fname.zz
		Parse each line in each file one by one as JSON, and yield it
	'''

	abet = string.ascii_lowercase
	numyielded = 0
	for s in (''.join(x) for x in combinations_with_replacement(string.ascii_lowercase, 2)):
		try:
			loadfn = fname+s
			with open(loadfn, 'r') as f:
				numyielded += 1
				#print "Loading {}".format(loadfn)
				for l in f.readlines():
					yield json.loads(l)
			
		except IOError as e:
			if numyielded == 0:
				raise Exception("No files matching {}.aa".format(fname))
			return