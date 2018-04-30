from flask import Flask,send_from_directory,request
import os
import tfidf_ir
import NB_ir

app = Flask(__name__)

method = 'tfidf'

@app.route('/tfidf')
def home():
	return send_from_directory('..','home.html')

@app.route('/tfidf')
def tfidf():
	method = 'tfidf'
	return send_from_directory('..','home.html')

@app.route('/nb')
def nb():
	method = 'nb'
	return send_from_directory('..','home.html')

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


