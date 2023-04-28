import pandas as pd
import numpy as np
from collections import OrderedDict
from collections import Counter

import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')



class PreProcess:

    def __init__(self,df):
        self.df = df

    def __call__(self):
        
        
        self.drop_column()
        self.clean_date()
        self.clean_star()
        self.clean_help()
        self.remove_duplicate_rows()
        self.drop_null_reviews()

        # review preprocessing
        self.extract_review()
        self.lowercase()
        self.remove_stopwords()
        self.remove_punctuations()
        self.remove_html_tags()
        self.remove_urls()
        self.remove_whitespaces()
        self.remove_duplicate_words()
        self.remove_frequent_words()
        self.remove_rare_words()
        self.remove_null_strings()
        self.tokenize()
        self.lemmatize()
        
    
        return self.df

    def input_preprocess(self):
        # self.lowercase()
        self.remove_stopwords()
        self.remove_punctuations()
        self.remove_html_tags()
        self.remove_urls()
        self.remove_whitespaces()
        self.remove_duplicate_words()
        self.remove_null_strings()
        self.tokenize()
        self.lemmatize()

        return self.df


    

    def drop_column(self):
        if 'Unnamed: 5' in self.df.columns:
            self.df = self.df.drop(["Unnamed: 5", "Unnamed: 6"], axis = 1)
    
    def clean_date(self):
        if 'date' in self.df.columns:
            self.df['date'] = self.df['date'].str.extract(r'on\s+(\d{1,2}\s+\w+\s+\d{4})', expand=False)
            self.df['date'] = pd.to_datetime(self.df['date'], format='%d %B %Y')

    def clean_star(self):
        if 'star' in self.df.columns:
            self.df['star']  = self.df['star'].str.extract(r'^([\d.]+)\s+out',expand=False)

    def clean_help(self):
        if 'help' in self.df.columns:
            self.df['help']  = self.df['help'].str.extract(r'^([\d]+)\s+people',expand=False)
            self.df['help']  = self.df['help'].fillna(0)

    def remove_duplicate_rows(self):
        self.df = self.df.drop_duplicates(subset=['title','star','review','date','help'])

    def drop_null_reviews(self):
        self.df = self.df.dropna()

    def extract_review(self):
        self.df = self.df['review']

    def lowercase(self):
        print("from lowercase",self.df.shape)
        self.df = self.df.str.lower()
    
    def remove_stopwords(self):
        self.stopwords = set(stopwords.words('english'))
        self.df = self.df.apply(lambda x: ' '.join([w for w in x.split() if w not in (self.stopwords)]))

    def remove_punctuations(self):
        self.df = self.df.str.replace(r'[^a-zA-Z0-9\s]', ' ')

    def remove_html_tags(self):
        self.df = self.df.str.replace(r'<[^<>]*>', ' ', regex=True)
    
    def remove_urls(self):
        self.df = self.df.str.replace(r'\s*https?://\S+(\s+|$)',' ')
    
    def remove_whitespaces(self):
        self.df = self.df.str.replace(r'^\s*|\s\s*',' ')
        self.df = self.df.str.strip()
    
    def remove_duplicate_words(self):
        self.df = (self.df.str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))
    
    def remove_frequent_words(self):        
        
        word_count = Counter()

        for text in self.df:
            for word in text.split():
                word_count[word] +=1

        word_count.most_common(20)
        frequent_words = set(word for (word,wc) in word_count.most_common(5))
        self.df = self.df.apply(lambda x: " ".join([word for word in x.split() if word not in frequent_words]))
    

    def remove_rare_words(self):
        word_count = Counter()

        for text in self.df:
            for word in text.split():
                word_count[word] +=1

        rare_words = set(word for (word,wc) in word_count.most_common()[:-100:-1])
        self.df = self.df.apply(lambda x: " ".join([word for word in x.split() if word not in rare_words]))


    def remove_null_strings(self):
        self.df.replace('', np.nan, inplace=True)
        self.df.dropna(inplace=True)

    
    def tokenize(self):
        self.df = self.df.apply(word_tokenize)
        

    def lemmatize(self):
        
        self.df = self.df.apply(lambda x:self.lemmatize_tokens(x))


    def lemmatize_tokens(self,tokens):
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

    
    

        
    


    







    

    





