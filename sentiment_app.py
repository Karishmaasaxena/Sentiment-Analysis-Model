import streamlit as st
from transformers import pipeline


def load_pipeline():
    return pipeline(task = "text-classification", model = "distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_pipeline()

st.title("sentiment Analysis App")

st.write("enter a sentence or phrase on sentiment analysis")

# inputs from the user

user_input = st.text_area("Enter text area")

# analyse sentimenr when user provide input

if st.button("Analyse sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)
        sentiment = result[0]["label"]
        confidence = result[0]['score']
        st.write(f"**Sentiment:**  {sentiment}")
        st.write(f"**confidence score:**  {confidence:.2f}")
    else:
        st.warning("please enter some text for analysis")