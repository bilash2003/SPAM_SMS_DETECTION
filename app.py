import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Streamlit UI
st.title("📩 SMS Spam Detection App")

input_sms = st.text_area("Enter the SMS text here")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚫 SPAM MESSAGE")
    else:
        st.success("✔ HAM (Not spam)")
