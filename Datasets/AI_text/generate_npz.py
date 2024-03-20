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

import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
import numpy as np
import tensorflow_hub as hub

def preprocess_data(df):

    # Load Universal Sentence Encoder from TensorFlow Hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    
    # Transform text data into 512-dimensional float representation
    def text_to_embedding(text):
        embedding = embed([text])[0]
        return embedding.numpy()
    
    # Apply transformation to each row in the DataFrame
    df['text_embedding'] = df['text'].apply(text_to_embedding)
    
    # Create DataFrame with 512 columns from the embedding vectors
    embedding_df = pd.DataFrame(df['text_embedding'].tolist(), columns=[f'feature_{i}' for i in range(512)])
    
    # Reset index of the original DataFrame to ensure proper alignment during concatenation
    df.reset_index(drop=True, inplace=True)

    # Concatenate the new DataFrame with the label column
    df = pd.concat([embedding_df, df['label']], axis=1)

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