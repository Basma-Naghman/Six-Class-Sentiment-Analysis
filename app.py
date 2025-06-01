import streamlit as st
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import re



#load the model
model = pickle.load(open('logistic_regression.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

# custom fuction 

stopword = set(nltk.corpus.stopwords.words("english"))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]"," ",text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopword]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    
    predicted_label = model.predict(input_vectorized)[0]
    
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    
    label = np.max(model.predict(input_vectorized)[0])
    
    return predicted_emotion,label
model
#app
st.title("Six NLP Sentiment Analysis app")
st.write(['Joy','Fear','love','Anger','Sadness','Surprise'])
input_text = st.text_input("Paste your text here")

if st.button("predict"):
    predicted_emotion,label = predict_emotion(input_text) 
    st.write("Predicted Emotion : ",predicted_emotion)
    st.write("Predicted Label :",label)


