from sklearn.feature_extraction.text import CountVectorizer
from model.splitdata import SplitData
from model.logistic_regression import LogRegress
import pickle

class Model:
    def __init__(self,df):
        self.df = df
    
    def __call__(self):
        x_text = self.df['review']
        y_text = self.df['label']

        # Split the data into training and testing sets with a 70-30 ratio
        splitdata = SplitData()
        x_text_train, x_text_test, y_text_train, y_text_test = splitdata.train_test_split(x_text, y_text)
        

        # Vectorize data
        cv = CountVectorizer(binary=True)
        cv.fit(x_text_train)

        # pickle the vectorizer
        with open('../pickle_files/vectorizer.pkl','wb') as f:
            pickle.dump(cv,f)

        x = cv.transform(x_text_train)
        x_test = cv.transform(x_text_test)        


        # Split the data into training and validation sets with a 70-30 ratio
        x_train, x_val, y_train,y_val = splitdata.train_val_split(x,y_text_train) 

        ### logistic regression ###
        
        log_reg = LogRegress(x_train, y_train,x_val, y_val)
        log_reg()

        # predict
        y_pred = log_reg.predict(x_test)

        # confusion matrix
        confusion_matrix, accuracy = log_reg.confusion_matrix(y_text_test,y_pred)
        print("Test Acccracy:",accuracy)
        print("Confusion matrix:",confusion_matrix)

        # pickle the model for deployment
        pickle.dump(log_reg,open('../pickle_files/reg_model.pkl','wb'))


        return y_text_test,y_pred


        


        










        
        