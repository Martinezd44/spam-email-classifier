import pandas as pd

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Keep label and text
data.columns = ['label', 'text']  # Rename
print("First 5 rows:")
print(data.head())
print("\nLabel counts:")
print(data['label'].value_counts())  # ham: ~4827, spam: ~747