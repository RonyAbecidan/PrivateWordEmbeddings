import pandas as pd
import numpy as np
import gensim
from gensim.models import KeyedVectors
import string 

def lower_case(text):
    '''@text : a simple string
       return a lowered version of the input string
    '''
    text=str(text)
    return text.lower()

def remove_punctuation(text):
    '''@text : a simple string
       return a version without punctuations of the input string
    '''
    return "".join([char for char in text if (char not in string.punctuation)])

vlower=np.vectorize(lower_case)
vpunc=np.vectorize(remove_punctuation)

def attack_efficiency(attacker_model,private_model,topn=3,display=False):
    '''@attacker_model : the embedding model of the attacker
       @private_model : the embedding model of the user
       @topn : the number of closest neighbour in terms of embeddings
       @display : a boolean enabling to display the words approximately recovered
       
       return the proportion of private word approximately recovered (belonging to the topn closest neighbour) by the attacker
    '''
    private_words=private_model.wv.vocab.keys()
    s=0
    for word in private_words:
        l=[]
        most_similar_words=attacker_model.similar_by_vector(private_model[word], topn=topn, restrict_vocab=None)
        for prop,_ in most_similar_words:
            l.append(prop)
            
        s+= int(word in l)
        if ((display) and (word in l)):
            print(word)
        
    return round(s/len(private_words),2)


def privacy_leak(private_sentence,attacker_model,private_model,display=True):
    '''@private_sentence : an input sentence presumably written by the user attacked
       @attack_model : the embedding model of the attacker
       @private_model : the embedding model of the user
       @display : a boolean enabling to display the words recovered using the closest neighbour  
       
       return the sentence recovered by the attacker (the most probable according to his model)
    '''
    words=private_sentence.lower().split(' ')
    predicted_sentence=''
    for word in words:
        prediction=attacker_model.similar_by_vector(private_model[word], topn=1, restrict_vocab=None)[0][0]
        predicted_sentence=predicted_sentence + prediction +' '
        confidence=round(attacker_model.similar_by_vector(private_model[word], topn=1, restrict_vocab=None)[0][1],2)
        if display:
            print(f' {prediction} | confidence: {confidence}')
        
    return predicted_sentence

def normalize(x):
    '''@x : a vector  
       return the normalization of x enabling to enforce its range into [0,1]
    '''
    return (x-np.min(x))/(np.max(x)-np.min(x))
        
def normalize_and_make_private(x,epsilon):
    '''@x : a vector  
       @epsilon : a parameter controling the amount of noise (eps-DP)
       return a noisy version of the normalization of x enabling to enforce its range into [0,1] (laplace noise(1/epsilon))
    '''
    x=normalize(x)+np.random.laplace(scale=1/epsilon,size=x.shape)
    return x

def normalize_embedding(model,hidden_dim=100):
    '''@model : an embedding model
       @hidden_dim : the dimension of the embedding space
       return the normalization of an embedded vocabulary to force its range belonging to [0,1]
    '''
    output_model = KeyedVectors(hidden_dim)
    vocab=model.wv.vocab.keys()
    for word in vocab:
        output_model.add(word,normalize(model[word]))
    
    return output_model

def private_embedding(model,epsilon,random_state=0,hidden_dim=100):
    '''@model : an embedding model
       @epsilon : a parameter controling the amount of noise (eps-DP)
       @random_state : a parameter controling the random state for reproducibility
       @hidden_dim : the dimension of the embedding space
       return a DP version of the embedded vocabulary inspired by the paper studied
    '''
    output_model = KeyedVectors(hidden_dim)
    np.random.seed(random_state)
    vocab=model.wv.vocab.keys()
    for word in vocab:
        output_model.add(word,normalize_and_make_private(model[word],epsilon))
    
    return output_model


