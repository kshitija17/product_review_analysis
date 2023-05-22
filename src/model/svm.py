# import
from sklearn.svm  import SVC

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

class SVM:
    def __init__(self,x_train,y_train,x_val,y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
    
    def __call__(self):
        for c in [0.01,0.5,1]:
            self.svm_clssifier = SVC(C=c, kernel = 'linear', random_state = 0)
            self.svm_clssifier.fit(self.x_train,self.y_train)
            print("Accuracy for c",c)
            print(accuracy_score(self.y_val,self.svm_clssifier.predict(self.x_val)))

    def predict(self,x_test):
        self.y_pred = self.svm_clssifier.predict(x_test)
        return self.y_pred
    
    def confusion_matrix(self,y_text_test,y_pred):
        # confsion matrix
        cm = confusion_matrix(y_text_test,y_pred,labels=[-1, 0,1])
        print(cm)
        accuracy = accuracy_score(y_text_test,y_pred)
        ax = sns.heatmap(cm,  xticklabels=['Negative', 'Neutral','Positive'], yticklabels=['Negative', 'Neutral','Positive'], annot=True, cmap='Blues', fmt='g')
        ax.set_ylabel('Predicted labels',fontsize=12, labelpad=15)
        ax.set_xlabel('Actual labels',fontsize=12, labelpad=15)
        plt.show()

        return cm, accuracy
    
    def metrics(self,y_text_test,y_pred):
        # y_true contains the true labels of the test set, and y_pred contains the predicted labels
        precision, recall, f1_score, support = precision_recall_fscore_support(y_text_test, y_pred)

        # precision, recall, and f1_score will be arrays containing the metrics for each class
        # support will be an array containing the number of samples for each class
        print('Class 0: Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}'.format(precision[0], recall[0], f1_score[0]))
        print('Class 1: Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}'.format(precision[1], recall[1], f1_score[1]))
        print('Class 2: Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}'.format(precision[2], recall[2], f1_score[2]))
    


    


    


