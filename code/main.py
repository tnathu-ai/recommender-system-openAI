import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from evaluation_utils import evaluate_model_predictions_rmse_mae
from CF_utils import pearson_correlation, get_rec_sys_directory
from path_utils import *
from matrix_factorization_utils import MFModel
from deep_learning_recommender import get_embeddings_deep_learning
from train_test_split_methods import random_split, sequential_split

def load_data(dataset_name):
    global USER_COLUMN_NAME, ITEM_ID_COLUMN, RATING_COLUMN_NAME, TITLE_COLUMN_NAME, TIMESTAMP_COLUMN_NAME, DATA_DIR

    DATA_DIR = get_rec_sys_directory('data')

    if dataset_name == 'amazon-beauty':
        data_path = os.path.join(DATA_DIR, 'amazon-beauty/large_merged_data.csv')
        USER_COLUMN_NAME = 'reviewerID'
        ITEM_ID_COLUMN = 'asin'
        RATING_COLUMN_NAME = 'rating'
        TITLE_COLUMN_NAME = 'title'
        TIMESTAMP_COLUMN_NAME = 'timestamp'
    elif dataset_name == 'ml-1m':
        data_path = os.path.join(DATA_DIR, 'ml-1m/ratings.dat')
        USER_COLUMN_NAME = 'UserID'
        ITEM_ID_COLUMN = 'MovieID'
        RATING_COLUMN_NAME = 'Rating'
        TITLE_COLUMN_NAME = 'Title'
        TIMESTAMP_COLUMN_NAME = 'Timestamp'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if dataset_name == 'amazon-beauty':
        data = pd.read_csv(data_path)
    elif dataset_name == 'ml-1m':
        data = pd.read_csv(data_path, sep='::', engine='python', names=[USER_COLUMN_NAME, ITEM_ID_COLUMN, RATING_COLUMN_NAME, TIMESTAMP_COLUMN_NAME])
    
    return data

def preprocess_data(data):
    interaction_matrix = pd.pivot_table(data, index=USER_COLUMN_NAME, columns=ITEM_ID_COLUMN, values=RATING_COLUMN_NAME).fillna(0)
    csr_interaction_matrix = csr_matrix(interaction_matrix.values)
    return csr_interaction_matrix, interaction_matrix

def compute_similarities(interaction_matrix, method):
    if method == 'pearson':
        return pearson_correlation(interaction_matrix)
    elif method == 'cosine':
        return cosine_similarity(interaction_matrix)
    else:
        raise ValueError(f"Unknown similarity computation method: {method}")

def split_data(data, method):
    if method == 'random':
        return random_split(data)
    elif method == 'sequential':
        return sequential_split(data)
    else:
        raise ValueError(f"Unknown train-test split method: {method}")

def main(args):
    # Load and preprocess data for a specific dataset
    data = load_data(args.dataset)
    csr_interaction_matrix, interaction_matrix = preprocess_data(data)
    
    # Compute similarities
    user_similarities = compute_similarities(csr_interaction_matrix, args.similarity)

    # matrix factorization, deep learning, etc., based on user preferences

    # Split data
    train_data, test_data = split_data(data, args.split)

    # Evaluation

#  python main.py --dataset ml-1m --split random --num_ratings_per_user 5 --num_main_user_ratings 10 --num_similar_users 5 --similarity cosine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run recommender system with configurable parameters.")
    parser.add_argument('--dataset', type=str, choices=['ml-1m', 'amazon-beauty'], required=True, help='Dataset to use (ml-1m or amazon-beauty)')
    parser.add_argument('--split', type=str, choices=['random', 'sequential'], default='random', help='Train-test split method')
    parser.add_argument('--num_ratings_per_user', type=int, default=1, help='Number of ratings per user')
    parser.add_argument('--num_main_user_ratings', type=int, default=4, help='Number of main user ratings')
    parser.add_argument('--num_similar_users', type=int, default=4, help='Number of similar users')
    parser.add_argument('--similarity', type=str, choices=['pearson', 'cosine'], default='pearson', help='Similarity measure to use')

    args = parser.parse_args()

    main(args)
