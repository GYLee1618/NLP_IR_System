import dependencies
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import math
import numpy as np


dependencies.try_dependencies()

stopWords = set(stopwords.words('english'))

#takes filename and splits it into lines
#returns list of lines
def getlines(fname):
	file = open(fname,"r")

	if file.mode == 'r':
	    lines = file.read().splitlines()
	else:
	    raise ValueError("Could not read file ",fname,"!")
	#print(lines)
	return lines

#takes filename and splits it into tokens
#returns list of tokens
def tokenize(fname):
	file = open(fname,"r")

	if file.mode == 'r':
	    contents = file.read()
	else:
	    raise ValueError("Could not read file ",fname,"!")

	soup = BeautifulSoup(contents,'html.parser')

	tokens = word_tokenize(soup.get_text())

	for i in range(0,len(tokens)):
		splittok = tokens[i].split('-')
		tokens[i] = splittok[0]
		if len(splittok) > 1:
			tokens.append(splittok[1])

	return tokens

#takes list of words and throws out stopwords and stems the remainder
#returns the stemmed remainder
def sanitize(tokens):
	clean_tokens = []

	for token in tokens:
		if token not in stopWords and (token.isalpha()):# or len(token) > 1 and token != '...'):
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
	smooth = 0.01
	def __init__(self,fname):

		self.docs = getlines(fname)

		self.counts = np.ones([len(self.docs),1])*freq_Mat.smooth
		self.words = {}
		self.vocab_size = 0
		self.wtotals = np.zeros([len(self.docs),1])
		self.DF = np.matrix([1])

		i = 0

		for doc in self.docs:
			print('training on',doc)
			worddoc = []
			tokens = sanitize(tokenize(doc))

			word_isThere = dict(zip(self.words.keys(),[0]*self.vocab_size))

			for token in tokens:
				if token not in self.words.keys():
					if self.words:
						self.counts = np.hstack([self.counts,np.ones([len(self.docs),1])*freq_Mat.smooth])
						self.DF = np.hstack([self.DF,np.matrix([1])])

					self.wtotals += freq_Mat.smooth	
					self.words[token] = self.vocab_size
					self.vocab_size += 1
					
					worddoc.append(token)
					word_isThere[token] = 1

				if not word_isThere[token]:
					self.DF[0,self.words[token]] += 1
					word_isThere[token] = 1

				self.counts[i,self.words[token]]+=1
				self.wtotals[i] += 1
			i += 1

		self.TF = np.log2(self.counts) - np.log2(self.wtotals)




