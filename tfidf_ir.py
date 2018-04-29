import numpy as np
import core_func
from nltk.tokenize import word_tokenize

class Searcher(core_func.freq_Mat):	
	def __init__(self,fname):
		super().__init__(fname)
		print('calculating weights')
		#print(self.DF)
		
	def query(self,query):
		clean_query = core_func.sanitize(word_tokenize(query))
		
		QF = np.ones([len(self.words),1])*core_func.freq_Mat.smooth
		max_QF = 0

		for word in clean_query:
			if word in self.words:
				QF[self.words[word]] += 1
				if QF[self.words[word]] > max_QF:
					max_QF = QF[self.words[word]]
		
		queryscore = np.multiply(.5*.5*QF/max_QF,np.log2(len(self.docs)/np.transpose(self.DF)))
		weights = np.multiply((self.counts),np.log2(len(self.docs)/self.DF))

		np.savetxt('weights.csv',weights,delimiter=",")
		
		scores = np.divide(np.matmul(nltkp.transpose(queryscore),np.transpose(weights)),np.multiply(np.diagonal(np.matmul(np.transpose(weights),weights)),np.diagonal((np.matmul(queryscore,np.transpose(queryscore))))))
		
		ranked_docs = sorted(range(len(scores)), key=lambda k: scores[k])
		
		for x in range(len(self.docs)-1,-1,-1):
			print(self.docs[ranked_docs[x]])		


tfidfsearch = Searcher("./cacm_small.files")
while True:
	q = input("Query: ")
	tfidfsearch.query(q)