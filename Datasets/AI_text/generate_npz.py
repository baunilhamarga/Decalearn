import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import sys
sys.path.append('../..')
from Scripts import statistics

import warnings
warnings.filterwarnings('ignore')

import tensorflow_hub as hub

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def preprocess_data(df):
    # Dropping columns that are not significant
    df = df[['text', 'prompt_name', 'label']]

    # Dropping NaN
    df = df.dropna()

    # Function to get embeddings for each row in the 'text' column
    def get_embeddings(text):
        return embed([text]).numpy().tolist()[0]

    # Overwrite the 'text' and 'prompt_name' columns with the embeddings
    df['text'] = df['text'].apply(get_embeddings)
    df['prompt_name'] = np.array(df['prompt_name'].apply(get_embeddings))

    # Calculate the average embedding for each sample
    def average_embedding(embeddings):
        if not embeddings:
            return np.zeros(len(embeddings[0]))  # Return zeros if embeddings is empty
        return np.mean(embeddings, axis=0)

    # Apply the average_embedding function to each row in the 'name' and 'prompt_name' columns
    df['text'] = df['text'].apply(average_embedding)
    df['prompt_name'] = df['prompt_name'].apply(average_embedding)

    return df

def generate_npz(dataset, file_name="AI_text"):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop('label', axis=1), dataset['label'], test_size=0.3, random_state=12227)

    # Export .npz
    np.savez_compressed(file_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

if __name__ == '__main__':
    # Load dataset
    dataset = read_csv("train_v2_drcat_02.csv")
    preprocessed_data = preprocess_data(dataset)
    generate_npz(preprocessed_data)
    data = np.load('AI_text.npz')
    statistics.print_statistics(data, classification=True)