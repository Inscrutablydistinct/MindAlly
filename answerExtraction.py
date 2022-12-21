from transformers import pipeline
import torch
def ansExtraction(pos,clean_ans,sentence):
    question_answer = pipeline("question-answering")
    context = clean_ans[pos]
    
    result= question_answer(question=sentence[0],context=context)
    return result
