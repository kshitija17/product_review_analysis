from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class LogRegress:
    def __init__(self,x_train,y_train,x_val,y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
    
    def __call__(self):
        for c in [0.01,0.5,1]:
            self.log_reg = LogisticRegression(C=c)
            self.log_reg.fit(self.x_train,self.y_train)
            print("Accuracy for c",c)
            print(accuracy_score(self.y_val,self.log_reg.predict(self.x_val)))

    def predict(self,x_test):
        self.y_pred = self.log_reg.predict(x_test)
        return self.y_pred
    
    def confusion_matrix(self,y_text_test,y_pred):
        cm = confusion_matrix(y_text_test,y_pred,labels=[-1, 0,1])
        accuracy = accuracy_score(y_text_test,y_pred)
        # print(self.log_reg.coef_)
        # print("intercept value",self.log_reg.intercept_)
        # print(self.log_reg.get_params())
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.show()
        

        return cm, accuracy
    

    


    


