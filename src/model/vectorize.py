from sklearn.feature_extraction.text import CountVectorizer


class Vectorize:
    def __init__(x_text_train):
        self.x_text_train = x_text_train
        self.x_text_test = x_text_test
    
    def __call__():
        cv = CountVectorizer(binary=True)
        cv.fit(x_text_train)
        x = cv.transform(x_text_train)
        x_test = cv.transform(x_text_test)

        return 