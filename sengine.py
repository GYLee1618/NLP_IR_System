from flask import Flask,send_from_directory,request
import os
import tfidf_ir
import NB_ir
from tfidf_ir import TF_IDFSearcher
from NB_ir import NBSearcher
import core_func
import pickle

tfidfsearch = TF_IDFSearcher()
NBsearch = NBSearcher()

tfidftrainedfile = input("tfidf trained file:")
trainedtfidf = open(tfidftrainedfile,"rb")
tfidfsearch = pickle.load(trainedtfidf)

trainedNBfile = input("Naive Bayes trained file:")
trainedNB = open(trainedNBfile,"rb")
NBsearch = pickle.load(trainedNB)

app = Flask(__name__)

method = 'tfidf'

@app.route('/')
def home():
	return send_from_directory('','home.html')

@app.route('/tfidf')
def tfidf():
	method = 'tfidf'
	return send_from_directory('','home.html')

@app.route('/nb')
def nb():
	method = 'nb'
	return send_from_directory('','home.html')

@app.route('/search', methods=['GET','POST'])
def search():
	print(request.form['query'])
	if method == 'tfidf':
		tfidfsearch.query(request.form['query'])
	else:
		NBsearch.query(request.form['query'])
	print(request.form['query'])
	return send_from_directory('','home.html')

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)	




