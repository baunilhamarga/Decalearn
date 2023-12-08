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
    # Dropping columns that are not significant or could be unethical to use for our future data exploration and predictions.
    # Dropping 'last_review' beacause it has ~20% NaN values
    preprocessed_data = df.drop(['id','host_name','host_id','last_review'], axis=1)
    # Replacing all NaN values in 'reviews_per_month' with 0
    preprocessed_data.fillna({'reviews_per_month':0}, inplace=True)
    preprocessed_data = preprocessed_data.dropna()
    # Use get_dummies to one-hot encode the "room_type", "neighbourhood_group" and 'neighbourhood' columns
    preprocessed_data = pd.get_dummies(preprocessed_data, columns=['room_type','neighbourhood_group','neighbourhood'], prefix=['room_type','neighbourhood_group','neighbourhood'])

    # Dropping NaN
    preprocess_data = preprocess_data.dropna()

    # Function to get embeddings for each row in the 'name' column
    def get_embeddings(text):
        return embed([text]).numpy().tolist()[0]

    # Overwrite the 'name' column with the embeddings
    preprocessed_data['name'] = preprocessed_data['name'].apply(get_embeddings)

    # Calculate the average embedding for each sample
    def average_embedding(embeddings):
        if not embeddings:
            return np.zeros(len(embeddings[0]))  # Return zeros if embeddings is empty
        return np.mean(embeddings, axis=0)

    # Apply the average_embedding function to each row in the 'name' column
    df['name'] = df['name'].apply(average_embedding)


    return preprocessed_data

def generate_npz(dataset, file_name="AB_NYC_2019"):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop('price', axis=1), dataset['price'], test_size=0.3, random_state=12227)

    # Export .npz
    np.savez_compressed(file_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

if __name__ == '__main__':
    # Load dataset
    dataset = read_csv("AB_NYC_2019.csv", header=0,parse_dates=[0])
    preprocessed_data = preprocess_data(dataset)
    generate_npz(preprocessed_data)
    data = np.load('AB_NYC_2019.npz', allow_pickle=True)
    statistics.print_statistics(data)