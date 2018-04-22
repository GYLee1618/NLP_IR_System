from nltk import download
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

def try_dependencies():
	try:
		wordnet.synsets('burp')
	except:
		download('wordnet')

	try:
		word_tokenize('burp')
	except:
		download('punkt')
		
	try:
		stopwords.words('english')
	except:
		download('stopwords')