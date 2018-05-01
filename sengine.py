from flask import Flask,send_from_directory,request,render_template
import os
import tfidf_ir
import NB_ir
from tfidf_ir import TF_IDFSearcher
from NB_ir import NBSearcher
import core_func
import pickle

tfidfsearch = TF_IDFSearcher()
NBsearch = NBSearcher()

tfidftrainedfile = input("tfidf trained file: ")
trainedtfidf = open(tfidftrainedfile,"rb")
tfidfsearch = pickle.load(trainedtfidf)

trainedNBfile = input("Naive Bayes trained file: ")
trainedNB = open(trainedNBfile,"rb")
NBsearch = pickle.load(trainedNB)

app = Flask(__name__)

method = 'nb'
links = []

@app.route('/')
def home():
	return send_from_directory('','home.html')

@app.route('/search', methods=['GET','POST'])
def search():
	if request.form['method']== 'tfidf':
		links = tfidfsearch.query(request.form['query'])
	elif request.form['method']== 'nb':
		links = NBsearch.query(request.form['query'])		
	return render_template('home.html',links = links)
@app.route('/corpus/<path:pathname>')
def goto(pathname):
	finalpath = '/corpus/'
	paths = pathname.split('/')
	for path in paths[0:len(paths)-1]:
		finalpath += path + '/'
	print(finalpath)
	print(paths[len(paths)-1])
	return send_from_directory('corpus',pathname)


if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)	




