import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

def predict_spam(text):
    clean = preprocess_text(text)
    features = vectorizer.transform([clean]).toarray()
    prediction = model.predict(features)[0]
    return "Spam" if prediction == 1 else "Ham"

# Tests
print(f"'Win a free iPhone now!' -> {predict_spam('Win a free iPhone now!')}")
print(f"'Hey, meet at 3pm?' -> {predict_spam('Hey, meet at 3pm?')}")