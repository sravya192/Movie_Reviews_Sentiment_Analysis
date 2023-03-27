import streamlit as st
import numpy as np
from string import punctuation
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
stop_words = stopwords.words("english")


#CLEAN TEXT
@st.cache
def clean_review(text, remove_stopwords=True, lemmatize_words=True):
    #cleaning text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)
    
    #Removing punctuations
    
    text = ''.join([c for c in text if c not in punctuation])
    
    #Converting to lowercase
    
    text=text.lower()
    
    #Removing stopwords
    
    if remove_stopwords:
        text=word_tokenize(text)
        text=[w for w in text if w not in stop_words]
        text = " ".join(text)
        
    #Lemmatization
    
    if lemmatize_words:
        text = word_tokenize(text) 
        lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    return text


#PREDICTION
@st.cache
def prediction(review):
    #cleandata
    cleaned_review = clean_review(review)
    
    #loading model and prediction
    model=joblib.load("logisticregression_model_pipeline.pkl")
    #making predictions
    res=model.predict([cleaned_review])
    return res

rev = input()
val=prediction(rev)