import pickle
from flask import Flask, request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
# from dataloader.dataloader import DataLoader
from dataloader.preprocess import PreProcess

app = Flask(__name__)

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
    vectorizer = pickle.load(open('../../pickle_files/vectorizer.pkl','rb'))
    transform_df = vectorizer.transform(clean_df) 

    # prediction
    model = pickle.load(open('../../pickle_files/reg_model.pkl','rb'))
    prediction = model.predict(transform_df)
    # print("prediction",prediction[0])

    return jsonify(str(prediction[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    # print("this is data:",data)

    # convert input data in dataframe
    df = pd.DataFrame(np.array(data).reshape(1,-1))

    #preprocess data
    preprocess = PreProcess(df[0])
    clean_df = preprocess.input_preprocess()

    # Vectorize data
    vectorizer = pickle.load(open('../../pickle_files/vectorizer.pkl','rb'))
    transform_df = vectorizer.transform(clean_df)

    # prediction
    model = pickle.load(open('../../pickle_files/reg_model.pkl','rb'))
    prediction = model.predict(transform_df)

    return render_template("index.html", prediction_output="The review is {}".format(prediction[0]))



if __name__ == "__main__":
    app.run(debug=True)  



