import pandas as pd

import interact
import negemotion
import textSimilarity
import answerExtraction
CSV = pd.read_csv("/Users/adityajain/Desktop/textSem/data.csv")


Ques = []
Ans = []

for i in range (0,len(CSV)):
  text1 = str(CSV.loc[i].at["question"])
  ans1 = str(CSV.loc[i].at["answer"])
  Ques.append(text1)
  Ans.append(ans1)
  

import re
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    txt = re.sub("http\S*\s", " ", txt)
    txt = re.sub("#\S*\s", " ", txt)
    txt = re.sub("W+", " ", txt)
    txt = re.sub("@\S*\s", " ", txt)
    txt = re.sub("<p>", " ", txt)
    return txt

clean_ques = []
clean_ans = []

for line in Ques:
  formatted = ""
  ls = line.split()
  length = len(ls)
  for a in range(0,length):
    formatted = formatted + " " + ls[a]
  clean_ques.append(clean_text(formatted))
        
for line in Ans:
  formatted = ""
  ls = line.split()
  length = len(ls)
  for b in range(0,length):
    formatted = formatted + " " + ls[b]
  clean_ans.append(clean_text(formatted))
  
#sentence = ["I really want to talk to someone about my thoughts and feelings but I can't."]

#print(negemotion.sentiment_vader(sentence))
#for i in range(0,100):
#    sentence = str(input(">> "))
    
def chatbot(sentence):
    if (negemotion.sentiment_vader(sentence)=="Negative"):
      textSimilarity.textSimilarity(sentence,clean_ques)
      return(answerExtraction.ansExtraction(pos,clean_ans,sentence))
    else :
      return(interact.interact(sentence))

