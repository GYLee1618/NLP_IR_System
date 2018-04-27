import numpy as np
import core_func
from nltk.tokenize import word_tokenize

class Searcher(core_func.freq_Mat):	
	def __init__(self,fname):
		super().__init__(fname)
		self.weights = np.multiply(self.TF,np.log(len(self.docs)/self.DF))
		
	def query(self,query):
		clean_query = core_func.sanitize(word_tokenize(query))
		QF = np.ones([len(self.docs),1])*core_func.freq_Mat.smooth
		max_freq = 0

		for word in clean_query:
			if word in self.words:
				QF[self.words.index(word)] += 1
				if QF[self.words.index(word)] > max_freq:
					max_freq = QF[self.words.index(word)]
		
		queryscore = np.multiply((.5+.5*QF/max_freq),np.log(len(self.docs)/self.DF))

		scores = np.dot(self.weights,queryscore)
		ranked_docs = sorted(range(len(s)), key=lambda k: scores[k])
		for x in range(0,results-1):
			print(ranked_docs.index(x))


tfidfsearch = Searcher("./cacm.files")
q = input("Query: ")
tfidfsearch.query(q)
