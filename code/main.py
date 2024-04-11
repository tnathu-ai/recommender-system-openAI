import numpy as np
import pandas as pd
import os
import sys
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from CF_utils import *
from evaluation_utils import *

# Constants
DATA_DIR = './data'
AMAZON_DATA_PATH = os.path.join(DATA_DIR, 'amazon-beauty/large_merged_data.csv')
MOVIELENS_DATA_PATH = os.path.join(DATA_DIR, 'movielens/ml-1m/ratings.dat')

# Command-line arguments parsing
parser = argparse.ArgumentParser(description='Run Collaborative Filtering on specified dataset.')
parser.add_argument('--dataset', type=str, default='amazon', choices=['amazon', 'movielens'],
                    help='The dataset to use: amazon or movielens')
parser.add_argument('--split_method', type=str, default='random', choices=['random', 'sequential'],
                    help='The method to split the data: random or sequential')
parser.add_argument('--num_ratings_per_user', type=int, default=1,
                    help='The number of ratings per user')
parser.add_argument('--num_main_user_ratings', type=int, default=4,
                    help='The number of main user ratings')
parser.add_argument('--num_similar_users', type=int, default=4,
                    help='The number of similar users')
parser.add_argument('--similarity_metric', type=str, default='pearson', choices=['pearson', 'cosine', 'jaccard'],
                    help='The similarity metric to use: pearson, cosine, or jaccard')
args = parser.parse_args()

# Set column names based on dataset
if args.dataset == 'amazon':
    USER_COLUMN_NAME = 'reviewerID'
    ITEM_COLUMN_NAME = 'asin'
    RATING_COLUMN_NAME = 'rating'
    TIMESTAMP_COLUMN_NAME = 'unixReviewTime'
    data_path = AMAZON_DATA_PATH
elif args.dataset == 'movielens':
    USER_COLUMN_NAME = 'UserID'
    ITEM_COLUMN_NAME = 'MovieID'
    RATING_COLUMN_NAME = 'Rating'
    TIMESTAMP_COLUMN_NAME = 'Timestamp'
    data_path = MOVIELENS_DATA_PATH

# Load data
data = pd.read_csv(data_path)
print(f"Loaded data from {data_path}")

# Create User-Item Interaction Matrix
interaction_matrix = pd.pivot_table(data, index=USER_COLUMN_NAME, columns=ITEM_COLUMN_NAME, values=RATING_COLUMN_NAME).fillna(0)
csr_interaction_matrix = csr_matrix(interaction_matrix.values)

# Compute similarity matrix
if args.similarity_metric == 'pearson':
    similarity_matrix = pearson_correlation(csr_interaction_matrix)
elif args.similarity_metric == 'cosine':
    similarity_matrix = cosine_similarity(csr_interaction_matrix.toarray())
elif args.similarity_metric == 'jaccard':
    similarity_matrix = jaccard_similarity(csr_interaction_matrix)

# Prediction and evaluation based on the specified split method
if args.split_method == 'random':
    results_df = predict_ratings_with_CF_and_save(data, similarity_matrix, 
                                                  split_method='random', 
                                                  num_ratings_per_user=args.num_ratings_per_user,
                                                  num_similar_users=args.num_similar_users, 
                                                  num_main_user_ratings=args.num_main_user_ratings)
elif args.split_method == 'sequential':
    results_df = predict_ratings_with_CF_and_save_sequential(data, similarity_matrix, 
                                                             split_method='sequential', 
                                                             num_ratings_per_user=args.num_ratings_per_user,
                                                             num_similar_users=args.num_similar_users, 
                                                             num_main_user_ratings=args.num_main_user_ratings)

# Evaluate predictions
evaluate_model_predictions_rmse_mae(results_df)

# python script_name.py --dataset movielens --split_method random --num_ratings_per_user 5 --num_main_user_ratings 10 --num_similar_users 5 --similarity_metric cosine
if __name__ == "__main__":
    main()
