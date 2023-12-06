import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from utils import *
from constants import *
from tenacity import retry, stop_after_attempt, wait_random_exponential
import random


# Create interaction matrix with a lower threshold
def create_interaction_matrix(df, user_col, item_col, rating_col, threshold=1):
    interactions = df.groupby([user_col, item_col])[rating_col].sum().unstack().reset_index().fillna(0).set_index(user_col)
    interactions = interactions.applymap(lambda x: 1 if x >= threshold else 0)
    
    user_mapper = dict(zip(np.unique(df[user_col]), list(range(df[user_col].nunique()))))
    item_mapper = dict(zip(np.unique(df[item_col]), list(range(df[item_col].nunique()))))

    user_inv_mapper = dict(zip(list(range(df[user_col].nunique())), np.unique(df[user_col])))
    item_inv_mapper = dict(zip(list(range(df[item_col].nunique())), np.unique(df[item_col])))

    return csr_matrix(interactions.values), user_mapper, item_mapper, user_inv_mapper, item_inv_mapper



def fit_knn_model(interaction_matrix, n_neighbors=4):
    """
    Fit the k-Nearest Neighbors model.

    Args:
        interaction_matrix (csr_matrix): User-item interaction matrix.
        n_neighbors (int): Number of neighbors to consider.

    Returns:
        NearestNeighbors: Trained kNN model.
    """
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors, n_jobs=-1)
    model_knn.fit(interaction_matrix)
    return model_knn

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



# source RMIT courses
def pearson_correlation(interaction_matrix):
    """
    Compute the Pearson Correlation Coefficient matrix for the user-item interaction matrix.

    Args:
    interaction_matrix (numpy.ndarray): A 2D array where rows represent users and columns represent items.
                                        The values in the matrix are the ratings given by users to items.

    Returns:
    numpy.ndarray: A 2D array representing the Pearson Correlation Coefficients between each pair of users.
    """

    # Get the number of users
    n_users = interaction_matrix.shape[0]

    # Initialize the Pearson Correlation matrix with zeros
    pearson_corr_matrix = np.zeros((n_users, n_users))

    # Small constant to avoid division by zero
    EPSILON = 1e-9

    # Iterate over each pair of users
    for i in range(n_users):
        for j in range(n_users):
            # Get the rating vectors for the current pair of users
            user_i_vec = interaction_matrix[i, :]
            user_j_vec = interaction_matrix[j, :]

            # Create masks for ratings greater than 0 (indicating rated items)
            mask_i = user_i_vec > 0
            mask_j = user_j_vec > 0

            # Find indices of corrated items (items rated by both users)
            corrated_index = np.intersect1d(np.where(mask_i), np.where(mask_j))

            # Skip if no items are corrated
            if len(corrated_index) == 0:
                continue

            # Compute the mean rating for each user over corrated items
            mean_user_i = np.mean(user_i_vec[corrated_index])
            mean_user_j = np.mean(user_j_vec[corrated_index])

            # Compute the deviations from the mean for each user
            user_i_sub_mean = user_i_vec[corrated_index] - mean_user_i
            user_j_sub_mean = user_j_vec[corrated_index] - mean_user_j

            # Calculate the components for Pearson correlation
            r_ui_sub_r_i_sq = np.square(user_i_sub_mean)
            r_uj_sub_r_j_sq = np.square(user_j_sub_mean)

            r_ui_sum_sqrt = np.sqrt(np.sum(r_ui_sub_r_i_sq))
            r_uj_sum_sqrt = np.sqrt(np.sum(r_uj_sub_r_j_sq))

            # Calculate Pearson correlation and handle division by zero
            sim = np.sum(user_i_sub_mean * user_j_sub_mean) / (r_ui_sum_sqrt * r_uj_sum_sqrt + EPSILON)

            # Store the similarity in the matrix
            pearson_corr_matrix[i, j] = sim

    return pearson_corr_matrix




def cosine_similarity_manual(interaction_matrix):
    """
    Compute the Cosine Similarity matrix for the user-item interaction matrix.

    Args:
    interaction_matrix (numpy.ndarray): A 2D array where rows represent users and columns represent items.
                                        The values in the matrix are the ratings given by users to items.

    Returns:
    numpy.ndarray: A 2D array representing the Cosine Similarities between each pair of users.
    """

    # Get the number of users
    n_users = interaction_matrix.shape[0]

    # Initialize the Cosine Similarity matrix with zeros
    cosine_sim_matrix = np.zeros((n_users, n_users))

    # Iterate over each pair of users
    for i in range(n_users):
        for j in range(n_users):
            # Get the rating vectors for the current pair of users
            user_i_vec = interaction_matrix[i, :]
            user_j_vec = interaction_matrix[j, :]

            # Compute the dot product between the two vectors
            dot_product = np.dot(user_i_vec, user_j_vec)

            # Compute the magnitude (norm) of each vector
            norm_i = np.linalg.norm(user_i_vec)
            norm_j = np.linalg.norm(user_j_vec)

            # Calculate cosine similarity (handling division by zero)
            if norm_i == 0 or norm_j == 0:
                # If a vector has magnitude 0, the similarity is set to 0
                similarity = 0
            else:
                similarity = dot_product / (norm_i * norm_j)

            # Store the similarity in the matrix
            cosine_sim_matrix[i, j] = similarity

    return cosine_sim_matrix

def format_similar_users_ratings(similar_users_ratings):
    if not isinstance(similar_users_ratings, dict):
        return ''

    formatted_ratings = []
    for user_id, ratings in similar_users_ratings.items():
        formatted_ratings.append(f"\n\n{user_id}:")
        for rating in ratings:
            formatted_ratings.append(f"+ {rating}")

    return '\n'.join(formatted_ratings)



def get_all_similar_users_ratings(data, 
                                  user_mapper, 
                                  user_inv_mapper, 
                                  model_knn, 
                                  interaction_matrix, 
                                  title_column_name='title', 
                                  user_column_name='userId'):
    all_similar_users_ratings = {}
    for user_id in data[user_column_name].unique():
        similar_users_ratings = []
        user_idx = user_mapper.get(user_id)
        if user_idx is None:
            continue

        distances, indices = model_knn.kneighbors(interaction_matrix[user_idx], n_neighbors=10)  # Adjust neighbors as needed

        for idx in indices.flatten():
            if idx == user_idx:
                continue

            similar_user_id = user_inv_mapper[idx]
            similar_user_data = data[data[user_column_name] == similar_user_id]
            sampled_ratings = similar_user_data.sample(n=min(5, len(similar_user_data)), random_state=42)  # Increased samples

            for _, row in sampled_ratings.iterrows():
                similar_users_ratings.append(f"{row[title_column_name]} ({row['rating']} stars)")

        all_similar_users_ratings[user_id] = similar_users_ratings

    return all_similar_users_ratings



# Function to check data sparsity
def check_data_sparsity(df, user_col, item_col):
    total_ratings = len(df)
    num_users = df[user_col].nunique()
    num_items = df[item_col].nunique()
    sparsity = 1 - (total_ratings / (num_users * num_items))
    print(f"Total Ratings: {total_ratings}, Number of Users: {num_users}, Number of Items: {num_items}, Sparsity: {sparsity}")


