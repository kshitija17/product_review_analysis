import numpy as numpy
from dataloader.dataloader import DataLoader
from dataloader.preprocess import PreProcess
from dataloader.labeldata import LabelData
from model.model import Model
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
    y_test,y_pred = model()
    # print("y_test",y_test.shape)
    # print("y_pred",y_pred.shape)
    plt.scatter(y_test,y_pred)
    plt.show()
    z=y_test-y_pred
    # print(z)




if __name__=="__main__":
    run()