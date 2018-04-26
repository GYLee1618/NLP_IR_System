import dependencies
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import math
import numpy as np

dependencies.try_dependencies()

stopWords = set(stopwords.words('english'))

#takes filename and splits it into lines
#returns list of lines
def getlines(fname):
	file = open(fname,"r")

	if file.mode == 'r':
	    lines = file.readlines()
	else:
	    raise ValueError("Could not read file ",fname,"!")

	lines = file.readlines();

	return lines

#takes filename and splits it into tokens
#returns list of tokens
def tokenize(fname):
	file = open(fname,"r")

	if file.mode == 'r':
	    contents = file.read()
	else:
	    raise ValueError("Could not read file ",fname,"!")

	tokens = word_tokenize(contents)

	return tokens

#takes list of words and throws out stopwords and stems the remainder
#returns the stemmed remainder
def sanitize(tokens):
	clean_tokens = []
	
	for token in tokens:
		if token not in stopWords and (token.isalnum() or len(token) > 1 and token != '...'):
			clean_tokens.append(SnowballStemmer("english").stem(token))
	
	return clean_tokens

#takes two words
#returns true if synonymous
#else returns false
def is_Synonym(word1, word2):
	for syn in wordnet.synsets(word1):
		for l in syn.lemmas():
			if word2 == l.name():
				return True
	return False

#takes list of tokens and basis
#generates and returns vector based on basis
def doc2vec(tokens,basis):
	vec = {}
	for token in tokens:
		for word in basis:
			if word not in vec.keys():
				vec[word] = 0
			if token == word:
				vec[word] += 1
	return vec


class freq_Mat:
	smooth = 1
	def __init__(self,fname):
		self.docs = getlines(fname)
		self.counts = np.ones([length(docs),1])*freq_Mat.smooth
		self.words = []
		self.wtotals = np.zeros([length(docs),1])
		
		for doc in self.docs
			worddoc = []
			tokens = sanitize(tokenize(doc))

			for token in tokens:
				if token not in words:
					if words:
						self.counts = np.hstack([self.counts,np.ones([length(docs),1])*freq_Mat.smooth])

					self.wtotals += freq_Mat.smooth	
					self.words.append(token)
					worddoc.append(token)

				self.counts[self.docs.index(doc),self.words.index(token)]+=1
				self.wtotals[self.docs.index(doc)] += 1

		self.DF = np.zeros([length(self.words),1])
		
		for x in range(0,length(self.words)-1):
			for y in range(0,length(self.docs)-1):
				if counts[y,x] is not freq_Mat.smooth:
					self.DF[x] += 1

		self.TF = np.log2(self.counts) - np.log2(self.wtotals)
		





