from flask import Flask, request, send_from_directory
import gunicorn
from sklearn.externals import joblib
import numpy as np
import scipy
import json

app = Flask(__name__)
vct = joblib.load('vct.pkl')
clf = joblib.load('RandomForest.pkl')


@app.route("/")
def hello():
    return "Hello World!"

@app.route("/yo")
def yo():
    return app.send_static_file('index.html')

@app.route('/getTags')
def getTags():
    title = request.args.get('title')
    #body = request.args.get('body')
    titles = [title]
    test_data_features = vct.transform(titles)
    test_data_features = test_data_features.toarray()
    return json.dumps(clf.predict(test_data_features).tolist())
if __name__ == "__main__":
    app.run()