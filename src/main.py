import numpy as numpy
from .dataloader.dataloader import DataLoader
from .dataloader.preprocess import PreProcess
from .dataloader.labeldata import LabelData
from .model.model import Model
import pickle

import matplotlib.pyplot as plt

def run():

    data_path = "/path"
    

    # load data
    data = DataLoader()
    df = data(data_path)

    # preprocess data 
    preprocess = PreProcess(df)
    clean_df = preprocess()

    # pickle the preprocessing
    # with open('../pickle_files/preprocess.pkl','wb') as f:
        # pickle.dump(preprocess,f)

    #label data
    labeldata = LabelData(clean_df)
    labelled_df =  labeldata()

    # train
    model = Model(labelled_df)
    model()

    # logistic regression
    y_test,y_pred = model.log_reg()

    # plt.scatter(y_test,y_pred)
    # plt.show()

    # svm
    y_test,y_pred = model.svm()


    # decision tree
    y_test,y_pred = model.decision_tree()

    # random forest
    y_test,y_pred = model.random_forest()



if __name__=="__main__":
    run()