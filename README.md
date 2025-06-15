# 💬 Six Emotion Sentiment Analysis App (NLP + Streamlit)

A simple and interactive NLP-powered Streamlit application that detects the **emotional tone** of input text. The model classifies text into one of six emotions:

- 😊 Joy  
- 😨 Fear  
- ❤️ Love  
- 😡 Anger  
- 😢 Sadness  
- 😲 Surprise

---

## 🚀 Features

- Built with **Streamlit** for a lightweight UI
- Preprocessed using **TF-IDF vectorization**
- Uses a **Logistic Regression model** for prediction
- Handles text cleaning, stopword removal, stemming
- Trained with labeled emotion data
- Supports six emotions classification

---

## 🧠 How It Works

1. **User inputs text**
2. Text is cleaned (stopwords removed, stemming applied)
3. Cleaned text is vectorized with TF-IDF
4. Pre-trained logistic regression model predicts the emotion
5. Result is shown on the Streamlit web UI

---

## 📁 File Structure

- `app.py` – Main Streamlit application
- `logistic_regression.pkl` – Trained classifier
- `label_encoder.pkl` – For decoding emotion labels
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer
- `nltk` – Used for preprocessing (stopwords, stemming)

---

## 📦 Requirements

```bash
streamlit
pandas
numpy
nltk
scikit-learn

