import pickle
from flask import Flask, request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

from .model.logistic_regression import LogRegress
from .model.svm import SVM
from .model.decision_tree import DecisionTree
from .model.random_forest import RandomForest
from .model.model import Model

import sys
# sys.path.append('../')
# from dataloader.dataloader import DataLoader
from .dataloader.preprocess import PreProcess
# from ..dataloader import preprocess


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

## load the model

# reg_model =  pickle.load(open('../../reg_model.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # print(data['review'])
    # convert input data in dataframe
    df = pd.DataFrame(np.array(data['review']).reshape(1,-1))

    #preprocess data
    # preprocess = pickle.load(open('../../pickle_files/preprocess.pkl','rb'))
    preprocess = PreProcess(df[0])
    clean_df = preprocess.input_preprocess()
    # print(clean_df[0])

    # Vectorize data
    vectorizer = pickle.load(open('pickle_files/vectorizer.pkl','rb'))
    transform_df = vectorizer.transform(clean_df) 

    # prediction

    if data['model']=='log_reg':
        model = pickle.load(open('pickle_files/reg_model.pkl','rb'))

    if data['model'] == 'svm':
        model = pickle.load(open('.pickle_files/svm.pkl','rb'))

    return jsonify(str(prediction[0]))



@app.route('/predict',methods=['POST'])
def predict():
    # data = [x for x in request.form.values()]
    review = request.form['review']
    algo = request.form['algorithm']

    model = None

    # convert input data in dataframe
    df = pd.DataFrame(np.array(review).reshape(1,-1))

    #preprocess data
    preprocess = PreProcess(df[0])
    clean_df = preprocess.input_preprocess()

    # Vectorize data
    vectorizer = pickle.load(open('pickle_files/vectorizer.pkl','rb'))
    transform_df = vectorizer.transform(clean_df)

    # prediction
    if algo =='log-reg':
        model = pickle.load(open('pickle_files/reg_model.pkl','rb'))

    if algo == 'svm':
        model = pickle.load(open('pickle_files/svm.pkl','rb'))  

    if algo == 'decision-tree':
        model = pickle.load(open('pickle_files/decision_tree.pkl','rb'))  

    if algo == 'random-forest':
        model = pickle.load(open('pickle_files/random_forest.pkl','rb'))    

    prediction = model.predict(transform_df)

    review_polarity = None
    if prediction[0] == -1:
        review_polarity = "Negative"
    
    if prediction[0] == 1:
        review_polarity = "Positive"

    if prediction[0] == 0:
        review_polarity = "Neutral"
       

    return render_template("index.html", prediction_output="The review is {}".format(review_polarity))



if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=3000)
