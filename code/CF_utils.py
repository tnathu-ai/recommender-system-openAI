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
def pearson_correlation(interaction_matrix, epsilon_constant = EPSILON_CONSTANT):
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
    EPSILON_CONSTANT = epsilon_constant

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
            sim = np.sum(user_i_sub_mean * user_j_sub_mean) / (r_ui_sum_sqrt * r_uj_sum_sqrt + EPSILON_CONSTANT)

            # Store the similarity in the matrix
            pearson_corr_matrix[i, j] = sim

    return pearson_corr_matrix


def item_pearson_correlation(interaction_matrix):
    """
    Compute the Pearson Correlation Coefficient matrix for the item-item interaction matrix with significance weighting.

    Args:
        interaction_matrix (2D numpy array): A matrix where rows represent users and columns represent items.
                                              The values in the matrix are the ratings given by users to items.

    Returns:
        numpy.ndarray: A 2D array representing the Pearson Correlation Coefficients between each pair of items.
    """
    n_items = interaction_matrix.shape[1]  # Number of items
    np_item_pearson_corr = np.zeros((n_items, n_items))  # Initialize the Pearson Correlation matrix

    print("Starting item-item Pearson Correlation computation...")

    for i, item_i_vec in enumerate(interaction_matrix.T):
        for j, item_j_vec in enumerate(interaction_matrix.T):

            # if i % 50 == 0 and j % 50 == 0:  # Reduce the frequency of print statements
            #     print(f"Processing correlation between items {i} and {j}")

            # Ratings co-rated by the current pair of items
            mask_i = item_i_vec > 0
            mask_j = item_j_vec > 0

            corrated_index = np.intersect1d(np.where(mask_i), np.where(mask_j))
            if len(corrated_index) == 0:
                # print(f"No corrated ratings for items {i} and {j}. Skipping...")
                continue

            mean_item_i = np.sum(item_i_vec) / (np.sum(np.clip(item_i_vec, 0, 1)) + EPSILON_CONSTANT)
            mean_item_j = np.sum(item_j_vec) / (np.sum(np.clip(item_j_vec, 0, 1)) + EPSILON_CONSTANT)

            # print(f"Mean rating for item {i}: {mean_item_i}, item {j}: {mean_item_j}")

            item_i_sub_mean = item_i_vec[corrated_index] - mean_item_i
            item_j_sub_mean = item_j_vec[corrated_index] - mean_item_j

            r_ui_sub_ri_sq = np.square(item_i_sub_mean)
            r_uj_sub_rj_sq = np.square(item_j_sub_mean)

            r_ui_sub_ri_sq_sum_sqrt = np.sqrt(np.sum(r_ui_sub_ri_sq))
            r_uj_sub_rj_sq_sum_sqrt = np.sqrt(np.sum(r_uj_sub_rj_sq))

            sim = np.sum(item_i_sub_mean * item_j_sub_mean) / (r_ui_sub_ri_sq_sum_sqrt * r_uj_sub_rj_sq_sum_sqrt + EPSILON_CONSTANT)

            weighted_sim = (min(len(corrated_index), DELTA_CONSTANT) / DELTA_CONSTANT) * sim

            # print(f"Correlation between item {i} and item {j}: {weighted_sim}")

            np_item_pearson_corr[i][j] = weighted_sim

    print("Item-item Pearson Correlation computation completed.")
    return np_item_pearson_corr


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