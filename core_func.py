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

