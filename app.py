import os
import nltk
from nltk.corpus import stopwords
import streamlit as st
import pickle
import numpy as np
from model import clean_text

# Download NLTK data once
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

setup_nltk()

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# Load model and vectorizer with proper error handling
try:
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
    
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except FileNotFoundError as e:
    st.error(f"⚠️ Error loading model files: {e}")
    st.stop()

st.markdown("<h1 style='text-align: center;'>💬 Sentiment Analysis Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Analyze whether text is Positive or Negative developed by AHMAD KHAN</p>", unsafe_allow_html=True)

st.write("---")

text = st.text_area("✍️ Enter your text here:", height=150)

if st.button("🔍 Analyze Sentiment"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        try:
            clean = clean_text(text)
            vec = vectorizer.transform([clean])

            prediction = model.predict(vec)[0]
            probabilities = model.predict_proba(vec)[0]

            labels = model.classes_

            st.write("---")

            if prediction == "positive":
                st.success(f"😊 Positive ({round(probabilities[1]*100,2)}%)")
            else:
                st.error(f"😠 Negative ({round(probabilities[0]*100,2)}%)")

            st.subheader("📊 Sentiment Confidence")

            chart_data = {
                "Sentiment": labels,
                "Probability": probabilities
            }

            st.bar_chart(chart_data, x="Sentiment", y="Probability")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")