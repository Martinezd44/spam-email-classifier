import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Handle NaN, non-strings, or empty inputs
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

# Load data
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Apply preprocessing
data['clean_text'] = data['text'].apply(preprocess_text)
print("Sample cleaned text:")
print(data[['text', 'clean_text']].head())

# Save cleaned data
data.to_csv('processed_data.csv', index=False)