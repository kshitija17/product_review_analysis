from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
import sys
import pickle

sys.path.append('../')
from dataloader.preprocess import PreProcess


class Predict:
    def __init__(self,input_review):
        self.input_review = input_review

    def __call__(self):

        # convert input strinf in pandas dataframe
        # df = pd.read_csv(pd.compat.StringIO(+"review\n"+input_review))


        df = pd.DataFrame(self.input_review)
        # df = pd.DataFrame(self.input_review)

        # preprocess input data
        preprocess = PreProcess(df[0])
        clean_df = preprocess.input_preprocess()
        print(clean_df[0])

        # Vectorize data
        vectorizer = pickle.load(open('../../pickle_files/vectorizer.pkl','rb'))
        transform_df = vectorizer.transform(clean_df) 

        model = pickle.load(open('../../pickle_files/reg_model.pkl','rb'))
        prediction = model.predict(transform_df)

        return prediction




def predict_main():

    review = "loved it"
    print(np.array(review).reshape(1,-1))
    predict = Predict(np.array(review).reshape(1,-1))
    

    prediction = predict()
    print("prediction: ",prediction[0])

if __name__=="__main__":
    predict_main()

