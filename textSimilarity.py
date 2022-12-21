#from sentence_transformers import SentenceTransformer, util
#import numpy as np
#model = SentenceTransformer('sentence-transformers/distilbert-base-nli-max-tokens')
## encode list of sentences to get their embeddings
#def textSimilarity(sentence,clean_ques):
#    for i in range (0,len(clean_ques)):
#      sentence.append(clean_ques[i])
#    embedding1 = model.encode(sentence, convert_to_tensor=True)
#    embedding2 = model.encode(clean_ques, convert_to_tensor=True)
#        # compute similarity scores of two embeddings
#    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
#
#    for i in range(len(sentence)):
#        for j in range(0,len(clean_ques)):
#            arr.append(cosine_scores[i][j].item())
#            if (j%100==0):
#                print(j)
#
#    pos = 0
#    for i in range(0,len(arr)):
#        if arr[i]>arr[pos]:
#          pos = i
#
#
#    return pos

from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def textSimilarity(sentence,clean_ques):

    model = SentenceTransformer('bert-base-nli-mean-tokens')

    sentence_embeddings = model.encode(clean_ques)
    sentence_embeddings1 = model.encode(sentence)
    arr2 = util.pytorch_cos_sim(sentence_embeddings1, sentence_embeddings)
    arr3=[]
    for i in range(1):
        for j in range(0,len(clean_ques)):
            arr3.append(arr2[i][j].item())
            if (j%100==0):
                print(j)
    pos = 0
    for i in range(0,len(arr3)):
      if arr3[i]>arr3[pos]:
        pos = i
    return pos

