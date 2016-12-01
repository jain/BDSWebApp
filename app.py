from flask import Flask, request, send_from_directory
import gunicorn
from sklearn.externals import joblib
import numpy as np
import scipy
import json

app = Flask(__name__)

clfs = [
    (joblib.load('Perceptron.pkl'), joblib.load('vecPer.pkl'), 'perceptron'),
    (joblib.load('bnb.pkl'),joblib.load('vecbnb.pkl'), 'bnb'),
    (joblib.load('bnb2.pkl'),joblib.load('vecbnb2.pkl'), 'bnb2'),
    (joblib.load('sgd.pkl'),joblib.load('vecsgd.pkl'), 'sgd'),
    (joblib.load('mnb.pkl'),joblib.load('vecmnb.pkl'), 'mnb'),
    ]


@app.route("/hello")
def hello():
    return "Hello World!"

@app.route("/")
def yo():
    return app.send_static_file('index.html')

@app.route('/getTags')
def getTags():
    input_data=request.args.get("text")
    all_tags=set()
    dic = {}
    for clf in clfs:
        test_data_features = clf[1].transform([input_data]).toarray()
        result = clf[0].predict(test_data_features)[0]
        all_tags.add(result)
        dic[clf[2]]= result
        print clf[2], result
    dic['all_tags'] = list(all_tags)
    return json.dumps(dic)


if __name__ == "__main__":
    app.run()

