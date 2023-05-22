from sklearn.feature_extraction.text import CountVectorizer
from .splitdata import SplitData
from .logistic_regression import LogRegress
from .svm import SVM
from .decision_tree import DecisionTree
from .random_forest import RandomForest
import pickle

class Model:
    def __init__(self,df):
        self.df = df
    
    def __call__(self):

        x_text = self.df['review']
        y_text = self.df['label']

        # Split the data into training and testing sets with a 70-30 ratio
        splitdata = SplitData()
        self.x_text_train, self.x_text_test, self.y_text_train, self.y_text_test = splitdata.train_test_split(x_text, y_text)
        

        # Vectorize data
        cv = CountVectorizer(binary=True)
        cv.fit(self.x_text_train)

        # pickle the vectorizer
        with open('./pickle_files/vectorizer.pkl','wb') as f:
            pickle.dump(cv,f)

        x = cv.transform(self.x_text_train)
        self.x_test = cv.transform(self.x_text_test)        


        # Split the data into training and validation sets with a 70-30 ratio
        self.x_train, self.x_val, self.y_train,self.y_val = splitdata.train_val_split(x,self.y_text_train) 


    ### logistic regression ###
    def log_reg(self):

        log_reg = LogRegress(self.x_train, self.y_train,self.x_val, self.y_val)
        log_reg()

        # predict
        y_pred = log_reg.predict(self.x_test)

        # confusion matrix
        confusion_matrix, accuracy = log_reg.confusion_matrix(self.y_text_test,y_pred)
        print("Test Acccracy:",accuracy)
        print("Confusion matrix:",confusion_matrix)

        # metrics
        log_reg.metrics(self.y_text_test,y_pred)

        # pickle the model for deployment
        pickle.dump(log_reg,open('./pickle_files/reg_model.pkl','wb'))

        return self.y_text_test,y_pred



    ### svm ###
    def svm(self):

        svm = SVM(self.x_train, self.y_train,self.x_val, self.y_val)
        svm()

        # predict
        y_pred = svm.predict(self.x_test)

        # confusion matrix
        confusion_matrix, accuracy = svm.confusion_matrix(self.y_text_test,y_pred)
        print("Test Acccracy:",accuracy)
        print("Confusion matrix:",confusion_matrix)

        # metrics
        svm.metrics(self.y_text_test,y_pred)

        # pickle the model for deployment
        pickle.dump(svm,open('./pickle_files/svm.pkl','wb'))

        return self.y_text_test,y_pred

    ### decision_tree ###
    def decision_tree(self):

        decision_tree = DecisionTree(self.x_train, self.y_train,self.x_val, self.y_val)
        decision_tree()

        # predict
        y_pred = decision_tree.predict(self.x_test)

        # confusion matrix
        confusion_matrix, accuracy = decision_tree.confusion_matrix(self.y_text_test,y_pred)
        print("Test Acccracy:",accuracy)
        print("Confusion matrix:",confusion_matrix)

        # metrics
        decision_tree.metrics(self.y_text_test,y_pred)


        # pickle the model for deployment
        pickle.dump(decision_tree,open('./pickle_files/decision_tree.pkl','wb'))

        return self.y_text_test,y_pred

        ### random_forest ###
    def random_forest(self):

        random_forest = RandomForest(self.x_train, self.y_train,self.x_val, self.y_val)
        random_forest()

        # predict
        y_pred = random_forest.predict(self.x_test)

        # confusion matrix
        confusion_matrix, accuracy = random_forest.confusion_matrix(self.y_text_test,y_pred)
        print("Test Acccracy:",accuracy)
        print("Confusion matrix:",confusion_matrix)

        # metrics
        random_forest.metrics(self.y_text_test,y_pred)


        # pickle the model for deployment
        pickle.dump(random_forest,open('./pickle_files/random_forest.pkl','wb'))

        return self.y_text_test,y_pred



        


        










        
        