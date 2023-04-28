from sklearn.model_selection import train_test_split


class SplitData:
    def __init__(self):
       pass
    
    def train_test_split(self,x_text,y_text):
        # x_text = self.df['review']
        # y_text = self.df['label']

        # Split the data into training and testing sets with a 70-30 ratio
        x_text_train, x_text_test, y_text_train, y_text_test = train_test_split(x_text, y_text, test_size=0.3, random_state=42)
      
        # row number
        x_text_train.reset_index(drop=True, inplace=True)
        x_text_test.reset_index(drop=True, inplace=True)
        y_text_train.reset_index(drop=True, inplace=True)
        y_text_test.reset_index(drop=True, inplace=True)
    
        return x_text_train, x_text_test, y_text_train, y_text_test
    
    def train_val_split(self,x,y):
        x_train, x_val, y_train,y_val = train_test_split(x,y, train_size = 0.8)

        return x_train, x_val, y_train,y_val
