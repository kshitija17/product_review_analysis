# import sentiment analyzer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import seaborn as sns
nltk.download('vader_lexicon')
import pandas as pd

class LabelData:

    def __init__(self,df):
        self.df = df

    def __call__(self):
        sia = SentimentIntensityAnalyzer()
        polarity_scores_dict = {}
        for index, row in tqdm(self.df.iteritems(), total=self.df.shape[0]):
            text = self.df[index]
            polarity_scores_dict[index] = sia.polarity_scores(text)
        
        # get labels from compound scores
        scores_df = pd.DataFrame(polarity_scores_dict).T
        scores_df['label']= scores_df['compound'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
        # scores_df.loc[(scores_df['pos'] >= scores_df['neu']) & (scores_df['pos'] >= scores_df['neg']), 'label'] = int(1)
        # scores_df.loc[(scores_df['neu'] > scores_df['pos']) & (scores_df['neu'] > scores_df['neg']), 'label'] = int(0)
        # scores_df.loc[(scores_df['neg'] > scores_df['pos']) & (scores_df['neg'] >= scores_df['neu']), 'label'] = int(-1)

        labelled_df = pd.concat([self.df,scores_df['compound'],scores_df['label']],axis=1)
        # labelled_df.shape
        labelled_df.reset_index(drop=True, inplace=True)
        # Create a scatterplot
        # sns.scatterplot(x=labelled_df.index, y='compound', hue='compound', data=labelled_df)
        # reset the numbers
        labelled_df = labelled_df.drop("compound", axis='columns')

        # oversampling 

        df_p = labelled_df[labelled_df['label']==1.0]       
        df_ne = labelled_df[labelled_df['label']==0.0]
        df_n = labelled_df[labelled_df['label']==-1.0]

        df_ne = df_ne.sample(df_p.shape[0],replace=True)
        df_n = df_n.sample(df_p.shape[0],replace=True)

        df = pd.concat([df_p,df_ne,df_n],axis=0)
    
        return df


