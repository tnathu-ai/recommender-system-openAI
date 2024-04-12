import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from tenacity import retry, stop_after_attempt, wait_random_exponential
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from utils import *
from constants import *
from scipy.sparse import csr_matrix

# Calculate Pearson Correlation Coefficient
# source: Yongli Ren. (2023) ' KNN_based_CF_final ' [python file], RMIT University, Melbourne
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


#--------------------------------------------

def weighted_pearson_correlation(interaction_matrix, epsilon_constant=EPSILON_CONSTANT, delta_constant=DELTA_CONSTANT):
    """
    Compute the weighted Pearson Correlation Coefficient matrix for the user-item interaction matrix.

    Args:
        interaction_matrix (numpy.ndarray): A dense array where rows represent users and columns represent items.
                                            The values in the array are the ratings given by users to items.
        epsilon_constant (float): A small constant to avoid division by zero.
        delta_constant (int): The threshold for significance weighting.

    Returns:
        numpy.ndarray: A 2D array representing the weighted Pearson Correlation Coefficients between each pair of users.
    """
    interaction_matrix = csr_matrix.toarray()
    n_users = interaction_matrix.shape[0]
    weighted_pearson_corr_matrix = np.zeros((n_users, n_users))

    for i in range(n_users):
        for j in range(n_users):
            user_i_vec = interaction_matrix[i, :]
            user_j_vec = interaction_matrix[j, :]

            mask_i = user_i_vec > 0
            mask_j = user_j_vec > 0
            corrated_index = np.intersect1d(np.where(mask_i)[0], np.where(mask_j)[0])

            if len(corrated_index) == 0:
                continue

            mean_user_i = np.mean(user_i_vec[corrated_index])
            mean_user_j = np.mean(user_j_vec[corrated_index])

            user_i_sub_mean = user_i_vec[corrated_index] - mean_user_i
            user_j_sub_mean = user_j_vec[corrated_index] - mean_user_j

            numerator = np.sum(user_i_sub_mean * user_j_sub_mean)
            denominator = np.sqrt(np.sum(np.square(user_i_sub_mean))) * np.sqrt(np.sum(np.square(user_j_sub_mean)))

            if denominator == 0:
                continue

            correlation = numerator / (denominator + epsilon_constant)
            
            # Apply significance weighting
            weight = min(len(corrated_index), delta_constant) / delta_constant
            weighted_pearson_corr_matrix[i, j] = correlation * weight

    return weighted_pearson_corr_matrix

# epsilon_constant and delta_constant should be predefined or passed as parameters
#weighted_user_pearson_corr = weighted_pearson_correlation(interaction_matrix, epsilon_constant=1e-9, delta_constant=DELTA_CONSTANT)

# For item-item correlation, transpose the interaction matrix before passing it to the function
#weighted_item_pearson_corr = weighted_pearson_correlation(interaction_matrix.T, epsilon_constant=1e-9, delta_constant=DELTA_CONSTANT)


#--------------------------------------------

def user_cosine_similarity(interaction_matrix):
    """
    Compute the Cosine Similarity matrix for the user-user interaction matrix.

    Args:
    interaction_matrix (csr_matrix): A sparse matrix where rows represent users and columns represent items.
                                     The values in the matrix are the ratings given by users to items.

    Returns:
    numpy.ndarray: A 2D array representing the Cosine Similarities between each pair of users.
    """
    # Convert sparse matrix to dense format for cosine similarity computation
    dense_matrix = interaction_matrix.toarray()

    # Compute cosine similarity
    cosine_sim_matrix = cosine_similarity(dense_matrix)

    return cosine_sim_matrix

def item_cosine_similarity(interaction_matrix):
    """
    Compute the Cosine Similarity matrix for the item-item interaction matrix.

    Args:
    interaction_matrix (csr_matrix): A sparse matrix where rows represent users and columns represent items.
                                     The values in the matrix are the ratings given by users to items.

    Returns:
    numpy.ndarray: A 2D array representing the Cosine Similarities between each pair of items.
    """
    # Convert sparse matrix to dense format for cosine similarity computation
    dense_matrix = interaction_matrix.toarray()

    # Compute cosine similarity on the transpose of the matrix to calculate item-item similarities
    cosine_sim_matrix = cosine_similarity(dense_matrix.T)

    return cosine_sim_matrix

# user_cos_sim = user_cosine_similarity(interaction_matrix)
# item_cos_sim = item_cosine_similarity(interaction_matrix)

#--------------------------------------------


def jaccard_similarity(matrix, axis=1):
    """
    Compute the Jaccard Similarity matrix for the interaction matrix.

    Args:
    matrix (csr_matrix): A sparse matrix where rows represent users or items.
    axis (int): Axis to compute the similarity on. 1 for user-user, 0 for item-item.

    Returns:
    numpy.ndarray: A 2D array representing the Jaccard Similarities.
    """
    # Ensure matrix is in dense format
    dense_matrix = matrix.toarray()
    
    # If computing item-item similarity, transpose the matrix
    if axis == 0:
        dense_matrix = dense_matrix.T

    # Initialize the Jaccard Similarity matrix
    size = dense_matrix.shape[0]
    jaccard_sim_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            intersection = np.logical_and(dense_matrix[i], dense_matrix[j]).sum()
            union = np.logical_or(dense_matrix[i], dense_matrix[j]).sum()
            jaccard_sim_matrix[i, j] = intersection / float(union) if union != 0 else 0

    return jaccard_sim_matrix

def user_jaccard_similarity(interaction_matrix):
    """
    Compute the Jaccard Similarity matrix for the user-user interaction matrix.

    Args:
    interaction_matrix (csr_matrix): A sparse matrix where rows represent users and columns represent items.

    Returns:
    numpy.ndarray: A 2D array representing the Jaccard Similarities between each pair of users.
    """
    return jaccard_similarity(interaction_matrix, axis=1)

def item_jaccard_similarity(interaction_matrix):
    """
    Compute the Jaccard Similarity matrix for the item-item interaction matrix.

    Args:
    interaction_matrix (csr_matrix): A sparse matrix where rows represent users and columns represent items.

    Returns:
    numpy.ndarray: A 2D array representing the Jaccard Similarities between each pair of items.
    """
    return jaccard_similarity(interaction_matrix, axis=0)

# user_jaccard_sim = user_jaccard_similarity(interaction_matrix)
# item_jaccard_sim = item_jaccard_similarity(interaction_matrix)

#--------------------------------------------


# Find Valid Neighbors
def get_valid_neighbors(pcc_matrix, threshold=0.6):
    valid_neighbors = {}
    for i, row in enumerate(pcc_matrix):
        valid_neighbors[i] = np.where(row > threshold)[0]
    return valid_neighbors