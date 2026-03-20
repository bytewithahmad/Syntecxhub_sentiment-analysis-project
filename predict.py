import pickle
from model import clean_text

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

while True:
    text = input("Enter text: ")
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)
    print("Sentiment:", result[0])