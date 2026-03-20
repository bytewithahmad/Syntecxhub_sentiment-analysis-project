import streamlit as st
import pickle
import numpy as np
from model import clean_text

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.markdown("<h1 style='text-align: center;'>💬 Sentiment Analysis Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Analyze whether text is Positive or Negative developed by AHMAD KHAN</p>", unsafe_allow_html=True)

st.write("---")

text = st.text_area("✍️ Enter your text here:", height=150)

if st.button("🔍 Analyze Sentiment"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        clean = clean_text(text)
        vec = vectorizer.transform([clean])

        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]

        labels = model.classes_   # ['negative', 'positive']

        st.write("---")

        # Show result
        if prediction == "positive":
            st.success(f"😊 Positive ({round(probabilities[1]*100,2)}%)")
        else:
            st.error(f"😠 Negative ({round(probabilities[0]*100,2)}%)")

        # 📊 Bar Chart
        st.subheader("📊 Sentiment Confidence")

        chart_data = {
            "Sentiment": labels,
            "Probability": probabilities
        }

        st.bar_chart(chart_data, x="Sentiment", y="Probability")