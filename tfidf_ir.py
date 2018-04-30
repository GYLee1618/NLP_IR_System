import numpy as np
import core_func
from nltk.tokenize import word_tokenize
import pickle

class TF_IDFSearcher(core_func.freq_Mat):	
	def __init__(self,fname=''):
		if fname is not '':
			super().__init__(fname)
			print('calculating weights')
	
		
	def query(self,query):
		clean_query = core_func.sanitize(word_tokenize(query))
		#print(clean_query)
		
		QF = np.zeros([len(self.words),1])*core_func.freq_Mat.smooth
		max_QF = 0

		for word in clean_query:
			if word in self.words:
				QF[self.words[word]] += 1
		
		queryscore = np.multiply(QF/len(clean_query),np.log2(len(self.docs)/np.transpose(self.DF)))
		weights = np.multiply(1+(self.TF),np.log2(len(self.docs)/self.DF))

		scores = np.zeros(len(self.docs))

		for i in range(0,len(self.docs)):
			scores[i] = (np.dot(weights[i,:],queryscore))
			#print(i+1,np.linalg.norm(weights[i,:]))
			
		
		#print(scores)

		score_index = np.argsort(scores)

		links = []

		for x in range(len(self.docs)-1,-1,-1):
			if scores[score_index[x]] < -len(self.docs):
				break
			links.append(self.docs[score_index[x]])
		return links

if __name__ == "__main__":
	trainedtfidf =  open("trainedtfidf.txt","wb")
	files = input("Corpus: ")
	close
	tfidfsearch = TF_IDFSearcher(files)
	pickle.dump(tfidfsearch,trainedtfidf)
# else:
# 	trainedtfidf = open("trainedtfidf.txt","rb")
# 	tfidfsearch= pickle.load(trainedtfidf)


 	# files = input("Corpus: ")
 	# tfidfsearch = Searcher(files)
 	# while True:
 	# 	q = input("Query: ")
 	# 	print(tfidfsearch.query(q))

