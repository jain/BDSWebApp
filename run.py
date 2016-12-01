from flask import Flask, request, send_from_directory
import gunicorn
from sklearn.externals import joblib
import numpy as np
import scipy
import json

app = Flask(__name__)
perceptron_vectorizer = joblib.load('classifiers/Perceptron/vectorizer.pkl')
perceptron_classifier = joblib.load('classifiers/Perceptron/Perceptron.pkl')

SGD_vectorizer = joblib.load('classifiers/SGD/vectorizer.pkl')
SGD_classifier = joblib.load('classifiers/SGD/SGD.pkl')

bnb1_vectorizer = joblib.load('classifiers/bnb1/vectorizer.pkl')
bnb1_classifier = joblib.load('classifiers/bnb1/bnb.pkl')

bnb2_vectorizer = joblib.load('classifiers/bnb2/vectorizer.pkl')
bnb2_classifier = joblib.load('classifiers/bnb2/bnb.pkl')

mnb_vectorizer = joblib.load('classifiers/mnb/vectorizer.pkl')
mnb_classifier = joblib.load('classifiers/mnb/NBMultinomial.pkl')



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
    # input_data = "i'd like check uploaded file image file (e.g png, jpg, jpeg, gif, bmp) another file. problem i'm using uploadify upload files, changes mime type gives 'text/octal' something mime type, matter file type upload...is way check uploaded file image apart checking file extension using php?.how check uploaded file image without mime type?"
    # input_data = "i'd like check uploaded file image file (e.g png, jpg, jpeg, gif, bmp) another file. problem i'm using uploadify upload files, changes mime type gives 'text/octal' something mime type, matter file type upload...is way check uploaded file image apart checking file extension using php?.how check uploaded file image without mime type?"

    test_data_features = perceptron_vectorizer.transform([input_data])
    test_data_features = test_data_features.toarray()
    perceptron_result = perceptron_classifier.predict(test_data_features)
    all_tags.add(perceptron_result[0])
    print perceptron_result[0]

    test_data_features = SGD_vectorizer.transform([input_data])
    test_data_features = test_data_features.toarray()
    SGD_result = SGD_classifier.predict(test_data_features)
    all_tags.add(SGD_result[0])
    print SGD_result[0]

    test_data_features = bnb1_vectorizer.transform([input_data])
    test_data_features = test_data_features.toarray()
    bnb1_result = bnb1_classifier.predict(test_data_features)
    all_tags.add(bnb1_result[0])
    print bnb1_result[0]

    test_data_features = bnb2_vectorizer.transform([input_data])
    test_data_features = test_data_features.toarray()
    bnb2_result = bnb2_classifier.predict(test_data_features)
    all_tags.add(bnb2_result[0])
    print bnb2_result[0]

    test_data_features = mnb_vectorizer.transform([input_data])
    test_data_features = test_data_features.toarray()
    mnb_result = mnb_classifier.predict(test_data_features)
    all_tags.add(mnb_result[0])
    print mnb_result[0]
    if "Undefined" in all_tags:
        all_tags.remove("Undefined")

    return json.dumps({"perceptron_result":perceptron_result[0],"SGD_result":SGD_result[0],"bnb1_result":bnb1_result[0],
            "bnb2_result":bnb2_result[0],"mnb_result":mnb_result[0],"all_tags":list(all_tags)})


if __name__ == "__main__":
    app.run()

