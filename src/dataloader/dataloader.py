import pandas as pd


class DataLoader:

    def __init__(self):
        pass
    
    def __call__(self,datapath):
        self.datapath = datapath
        self.df = pd.read_csv('../data/dataset.csv')
    
        return self.df




