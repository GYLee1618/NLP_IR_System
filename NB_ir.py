import numpy as np
import core_func
from nltk.tokenize import word_tokenize
import pickle

class NBSearcher(core_func.freq_Mat):	
	def __init__(self,fname=''):
		if fname is not '':
			super().__init__(fname)
			print('calculating weights')

	def query(self,query):
		clean_query = core_func.sanitize(word_tokenize(query))

		scores = np.zeros(len(self.docs))

		for qtoken in clean_query:
			if qtoken in self.words:
				scores += self.TF[:,self.words[qtoken]]

		score_index = np.argsort(scores)

		links = []

		for x in range(len(self.docs)-1,-1,-1):
			if scores[score_index[x]] < -np.sqrt(len(self.docs)):
				break
			links.append(self.docs[score_index[x]])
		return links



if __name__ == "__main__":
	NBtrainedfile = input("File to save trained data to: ")
	trainedNB =  open(NBtrainedfile,"wb")
	files = input("Corpus: ")
	NBsearch = NBSearcher(files)
	pickle.dump(NBsearch,trainedNB)


