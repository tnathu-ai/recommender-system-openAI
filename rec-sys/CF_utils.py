import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from utils import *
from constants import *
from tenacity import retry, stop_after_attempt, wait_random_exponential



def create_interaction_matrix(df, user_col, item_col, rating_col, threshold=0):
    """
    Create a user-item interaction matrix.

    Args:
        df (DataFrame): DataFrame containing user-item interactions.
        user_col (str): Name of the user column.
        item_col (str): Name of the item column.
        rating_col (str): Name of the rating column.
        threshold (float): Minimum rating to consider for interaction.

    Returns:
        tuple: A tuple containing the interaction matrix and mapper dictionaries.
    """
    interactions = df.groupby([user_col, item_col])[rating_col].sum().unstack().reset_index().fillna(0).set_index(user_col)
    interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    
    user_mapper = dict(zip(np.unique(df[user_col]), list(range(df[user_col].nunique()))))
    item_mapper = dict(zip(np.unique(df[item_col]), list(range(df[item_col].nunique()))))

    user_inv_mapper = dict(zip(list(range(df[user_col].nunique())), np.unique(df[user_col])))
    item_inv_mapper = dict(zip(list(range(df[item_col].nunique())), np.unique(df[item_col])))

    X = csr_matrix(interactions.values)

    return X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper

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
