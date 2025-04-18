import streamlit as st
import pickle
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


ps = PorterStemmer()

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    text = word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    cleaned_tokens = []
    for token in text:
        if token.isalnum():
            cleaned_tokens.append(token)
    
    # Remove stopwords and punctuation
    filtered_tokens = []
    for token in cleaned_tokens:
        if token not in stopwords.words("english") and token not in string.punctuation:
            filtered_tokens.append(token)
    
    # Stemming
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_tokens.append(ps.stem(token))
    
    # Join tokens back into a single string
    return " ".join(stemmed_tokens)

# Load the TF-IDF vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# Load the model
@st.cache_resource
def load_model():
    with open('voting_classifier_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Streamlit UI
st.title("Spam Message Classifier")
st.write("This app classifies whether a given message is spam or not.")
st.subheader("Enter the message below:")
input_message = st.text_area("Message")

# Add predict button
if st.button("Predict"):
    if input_message:
        # Process the input message
        processed_text = transform_text(input_message)
        
        # Vectorize the processed text
        vectorized_text = tfidf.transform([processed_text])
        
        # Convert sparse matrix to dense array - this fixes the ValueError
        dense_vector = vectorized_text.toarray()
        
        # Make prediction with dense data
        prediction = model.predict(dense_vector)[0]
        
        # Display the result
        if prediction == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
            
        # Show prediction probabilities
        proba = model.predict_proba(dense_vector)[0]
        st.write(f"Confidence scores - Not Spam: {proba[0]:.4f}, Spam: {proba[1]:.4f}")
    else:
        st.warning("Please enter a message to classify.")

st.write("This app uses a machine learning model to classify messages as spam or not spam. The model is trained on a dataset of labeled messages and uses various features to make predictions. The app preprocesses the input message, vectorizes it using TF-IDF, and then uses the trained model to predict the class of the message.")