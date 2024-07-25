import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

# Specify the delimiter as a tab character
df = pd.read_csv("C:\\Users\\saipr\\Downloads\\Restaurant_Reviews.tsv", delimiter='\t')
# Split the data into features and target
X = df['Review']
y = df['Liked']
#Text Preprocessing
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def Preprocessing_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
X= df['Review'].apply(Preprocessing_text)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)
from sklearn.metrics import accuracy_score, classification_report  # Import evaluation metrics

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Define the new review
new_review = ["The food was bad and the service was worest!"]

# Convert the new review to numerical format using the previously fitted TF-IDF vectorizer
new_review_tfidf = vectorizer.transform(new_review)

# Predict the sentiment using the trained model
prediction = model.predict(new_review_tfidf)

# Output the prediction
print("Prediction (1 = Liked, 0 = Not Liked):", prediction[0])
import streamlit as st
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load('model.pkl')

def preprocess_text(text):
    # Convert text into the format expected by the model
    return vectorizer.transform([text])  # Transform the text into TF-IDF features

# Streamlit app
st.title('Restaurant Review Sentiment Analysis')
st.write('Enter a review to classify it as positive or negative')

# User input
user_input = st.text_area('Review')

if st.button('Classify'):
    if user_input:
        # Preprocess and convert the input text to numeric features
        preprocessed_input = preprocess_text(user_input)
        
        # Make prediction
        prediction = model.predict(preprocessed_input)
        
        # Assuming the model outputs binary classification
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        # Display the result
        st.write(f'Sentiment: {sentiment}')
    else:
        st.write('Please enter a review')




