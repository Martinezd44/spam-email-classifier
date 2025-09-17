# Spam Email Classifier

## Overview
A machine learning model to classify text messages as spam or ham with 96% accuracy, built with Python, scikit-learn, and NLTK.

## Features
- Preprocesses text using NLTK (removes punctuation, stopwords).
- Trains a Logistic Regression model with TF-IDF vectorization.
- Achieves 96% accuracy on the UCI SMS Spam Collection dataset.
- Visualizes performance with a confusion matrix.

## Dataset
- [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,572 messages labeled as spam or ham.

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/Martinezd44/spam-email-classifier.git
   cd spam-email-classifier
2. Create virtual environment:  
python -m venv env
env\Scripts\activate  # Windows

3. Install dependencies:
pip install pandas==2.2.3 scikit-learn==1.5.2 nltk==3.9.1 seaborn==0.13.2 matplotlib==3.9.2 joblib==1.4.2 numpy==2.1.1

4. Download NLTK data:
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

5. Download spam.csv from Kaggle and place in folder

Usage

Run preprocessing: python preprocess.py
Train model: python train_model.py
Test predictions: python predict.py

Results

Accuracy: ~96%.
See confusion_matrix.png for performance visualization.
spam-email-classifier/
├── spam.csv              # Dataset (download from Kaggle)
├── processed_data.csv    # Preprocessed data
├── preprocess.py         # Text preprocessing
├── train_model.py        # Model training
├── predict.py            # Prediction function
├── model.pkl             # Trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── confusion_matrix.png  # Performance visualization
├── README.md             # This file
Email: [martinezd44@montclair.edu]