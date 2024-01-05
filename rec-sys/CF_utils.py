import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from utils import *
from constants import *
from tenacity import retry, stop_after_attempt, wait_random_exponential
import random



# Calculate Pearson Correlation Coefficient
# source RMIT courses
def pearson_correlation(interaction_matrix):
    """
    Compute the Pearson Correlation Coefficient matrix for the user-item interaction matrix.

    Args:
    interaction_matrix (csr_matrix): A sparse matrix where rows represent users and columns represent items.
                                     The values in the matrix are the ratings given by users to items.

    Returns:
    numpy.ndarray: A 2D array representing the Pearson Correlation Coefficients between each pair of users.
    """
    # Convert sparse matrix to dense format for processing
    dense_matrix = interaction_matrix.toarray()
    
    # Get the number of users
    n_users = dense_matrix.shape[0]

    # Initialize the Pearson Correlation matrix
    pearson_corr_matrix = np.zeros((n_users, n_users))

    # Small constant to avoid division by zero
    EPSILON = 1e-9

    # Iterate over each pair of users
    for i in range(n_users):
        for j in range(n_users):
            # Get the rating vectors for the current pair of users
            user_i_vec = dense_matrix[i, :]
            user_j_vec = dense_matrix[j, :]

            # Masks for rated items
            mask_i = user_i_vec > 0
            mask_j = user_j_vec > 0

            # Find indices of corrated items
            corrated_index = np.intersect1d(np.where(mask_i)[0], np.where(mask_j)[0])

            # Skip if no items are corrated
            if len(corrated_index) == 0:
                continue

            # Compute the mean rating for each user over corrated items
            mean_user_i = np.mean(user_i_vec[corrated_index])
            mean_user_j = np.mean(user_j_vec[corrated_index])

            # Compute the deviations from the mean
            user_i_sub_mean = user_i_vec[corrated_index] - mean_user_i
            user_j_sub_mean = user_j_vec[corrated_index] - mean_user_j

            # Calculate the components for Pearson correlation
            r_ui_sub_r_i_sq = np.square(user_i_sub_mean)
            r_uj_sub_r_j_sq = np.square(user_j_sub_mean)

            r_ui_sum_sqrt = np.sqrt(np.sum(r_ui_sub_r_i_sq))
            r_uj_sum_sqrt = np.sqrt(np.sum(r_uj_sub_r_j_sq))

            # Calculate Pearson correlation
            sim = np.sum(user_i_sub_mean * user_j_sub_mean) / (r_ui_sum_sqrt * r_uj_sum_sqrt + EPSILON)

            # Store the similarity in the matrix
            pearson_corr_matrix[i, j] = sim

    return pearson_corr_matrix


def item_pearson_correlation(interaction_matrix):
    """
    Compute the Pearson Correlation Coefficient matrix for the item-item interaction matrix.

    Args:
    interaction_matrix (csr_matrix): A sparse matrix where rows represent users and columns represent items.
                                     The values in the matrix are the ratings given by users to items.

    Returns:
    numpy.ndarray: A 2D array representing the Pearson Correlation Coefficients between each pair of items.
    """
    # Convert sparse matrix to dense format for processing
    dense_matrix = interaction_matrix.toarray()
    n_items = dense_matrix.shape[1]

    # Initialize the Pearson Correlation matrix
    pearson_corr_matrix = np.zeros((n_items, n_items))
    EPSILON = 1e-9

    # Iterate over each pair of items
    for i in range(n_items):
        for j in range(n_items):
            item_i_vec = dense_matrix[:, i]
            item_j_vec = dense_matrix[:, j]

            mask_i = item_i_vec > 0
            mask_j = item_j_vec > 0

            corrated_index = np.intersect1d(np.where(mask_i)[0], np.where(mask_j)[0])

            if len(corrated_index) == 0:
                continue

            mean_item_i = np.mean(item_i_vec[corrated_index])
            mean_item_j = np.mean(item_j_vec[corrated_index])

            item_i_sub_mean = item_i_vec[corrated_index] - mean_item_i
            item_j_sub_mean = item_j_vec[corrated_index] - mean_item_j

            r_ui_sub_r_i_sq = np.square(item_i_sub_mean)
            r_uj_sub_r_j_sq = np.square(item_j_sub_mean)

            r_ui_sum_sqrt = np.sqrt(np.sum(r_ui_sub_r_i_sq))
            r_uj_sum_sqrt = np.sqrt(np.sum(r_uj_sub_r_j_sq))

            sim = np.sum(item_i_sub_mean * item_j_sub_mean) / (r_ui_sum_sqrt * r_uj_sum_sqrt + EPSILON)

            pearson_corr_matrix[i, j] = sim

    return pearson_corr_matrix




def recommend_items(user_id, interaction_matrix, user_mapper, item_inv_mapper, model_knn, n_recommendations=4):
    """
    Recommend items for a given user based on kNN model.

    Args:
        user_id (str): User ID for whom to make recommendations.
        interaction_matrix (csr_matrix): User-item interaction matrix.
        user_mapper (dict): Dictionary mapping user ID to user index.
        item_inv_mapper (dict): Dictionary mapping item index to item ID.
        model_knn (NearestNeighbors): Trained kNN model.
        n_recommendations (int): Number of recommendations to make.

    Returns:
        list: List of recommended item IDs.
    """
    user_idx = user_mapper[user_id]
    distances, indices = model_knn.kneighbors(interaction_matrix[user_idx], n_neighbors=n_recommendations+1)
    
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    recommendations = [item_inv_mapper[idx] for idx, dist in raw_recommends if idx != user_idx]

    return recommendations


# Function to check data sparsity
def check_data_sparsity(df, user_col, item_col):
    total_ratings = len(df)
    num_users = df[user_col].nunique()
    num_items = df[item_col].nunique()
    sparsity = 1 - (total_ratings / (num_users * num_items))
    print(f"Total Ratings: {total_ratings}, Number of Users: {num_users}, Number of Items: {num_items}, Sparsity: {sparsity}")



# Find Valid Neighbors
def get_valid_neighbors(pcc_matrix, threshold=0.6):
    valid_neighbors = {}
    for i, row in enumerate(pcc_matrix):
        valid_neighbors[i] = np.where(row > threshold)[0]
    return valid_neighbors