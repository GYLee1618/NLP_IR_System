import numpy as np
import core_func
from nltk.tokenize import word_tokenize

class Searcher(core_func.freq_Mat):	
	def __init__(self,fname):
		super().__init__(fname)
		print('calculating weights')

	def query(self,query):
		clean_query = core_func.sanitize(word_tokenize(query))

		scores = np.zeros(len(self.docs))

		for qtoken in clean_query:
			if qtoken in self.words:
				scores += self.TF[:,self.words[qtoken]]

		score_index = np.argsort(scores)

		for x in range(len(self.docs)-1,-1,-1):
			if scores[score_index[x]] < -12:
				break
			print(self.docs[score_index[x]], '\tscore:\t',scores[score_index[x]])	

files = input("Corpus: ")
NBsearch = Searcher(files)
while True:
	q = input("Query: ")
	NBsearch.query(q)
