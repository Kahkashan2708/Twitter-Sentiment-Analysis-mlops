import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def create_features(input_file, output_dir, max_features=10000):
    print("Loading processed data...")
    df = pd.read_csv(input_file)

    # Remove rows with missing or empty processed_text
    df = df[df['processed_text'].notnull()]
    df = df[df['processed_text'].str.strip() != '']

    X_text = df['processed_text']
    y = df['target']

    print("Generating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(X_text)

    os.makedirs(output_dir, exist_ok=True)

    # Save features, labels, and vectorizer
    with open(f"{output_dir}/features.pkl", "wb") as f:
        pickle.dump(X, f)
    with open(f"{output_dir}/labels.pkl", "wb") as f:
        pickle.dump(y, f)
    with open(f"{output_dir}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Feature engineering complete! Features shape: {X.shape}")
    return X, y, vectorizer

if __name__ == "__main__":
    input_file = "data/processed/clean_tweets.csv"
    output_dir = "data/processed"
    create_features(input_file, output_dir)
