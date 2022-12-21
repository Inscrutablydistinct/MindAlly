import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CSV = pd.read_csv("/Users/adityajain/Desktop/textSem/data.csv")
#calculate the negative, positive, neutral and compound scores, plus verbal evaluation
def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.1 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
  
    return overall_sentiment
    
#count = 0
#for i in range(2000,3000):
#  text = str(CSV.loc[i].at["question"])
#  emotion = sentiment_vader(text)
#  CSV.loc[i, 'Emotion'] = emotion
#  CSV.to_csv("/Users/adityajain/Desktop/textSem/data.csv", index=False)
#  if (i%100==0):
#    print(i)
#
    

