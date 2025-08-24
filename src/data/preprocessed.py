import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('stopwords', quiet=True)

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs/mentions/hashtags/special chars."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_data(input_file: str, output_file: str):
    """Load raw CSV without headers, clean and stem text, and save processed CSV."""
    # Define column names for the raw dataset
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']

    print("Loading raw data...")
    df = pd.read_csv(
        input_file,
        encoding='ISO-8859-1',
        header=None,
        names=columns,
        usecols=['target', 'text']
    )

    # Map labels: 4 → 1 (positive), 0 remains 0 (negative)
    print("Mapping labels...")
    df['target'] = df['target'].replace(4, 1)

    # Clean raw text
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Remove stopwords and apply stemming
    print("Removing stopwords and stemming...")
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def process_words(text: str) -> str:
        tokens = text.split()
        filtered = [stemmer.stem(tok) for tok in tokens if tok not in stop_words and len(tok) > 2]
        return ' '.join(filtered)

    df['processed_text'] = df['clean_text'].apply(process_words)

    # Keep only the target and processed text columns
    df_final = df[['target', 'processed_text']].copy()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to CSV
    print(f"Saving preprocessed data to {output_file} ...")
    df_final.to_csv(output_file, index=False)
    print(f"Preprocessing complete: {len(df_final)} rows saved.")

if __name__ == "__main__":
    raw_path = "data/raw/tweets.csv"
    proc_path = "data/processed/clean_tweets.csv"
    preprocess_data(raw_path, proc_path)