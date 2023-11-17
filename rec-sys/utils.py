import numpy as np
import openai
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import re
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from constants import *
from constants import *
from evaluation_utils import *

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY
AMAZON_CONTENT_SYSTEM = "Amazon Beauty products critic"

# Configured to retry up to STOP_AFTER_N_ATTEMPTS with an exponential backoff delay
retry_decorator = retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(STOP_AFTER_N_ATTEMPTS))

# Tokenizer setup for text processing
TOKENIZER = tiktoken.get_encoding(EMBEDDING_ENCODING)

def check_and_reduce_length(text, max_tokens=MAX_TOKENS_CHAT_GPT, tokenizer=TOKENIZER):
    """
    Check and reduce the length of the text to be within the max_tokens limit.

    Args:
        text (str): The input text.
        max_tokens (int): Maximum allowed tokens.
        tokenizer: The tokenizer used for token counting.

    Returns:
        str: The text truncated to the max_tokens limit.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_text = ''
    for token in tokens[:max_tokens]:
        truncated_text += tokenizer.decode([token])

    return truncated_text

def extract_numeric_rating(rating_text):
    """
    Extract numeric rating from response text.

    Args:
        rating_text (str): Text containing numeric rating.

    Returns:
        float: Extracted rating value. Returns 0 for unexpected responses.
    """
    try:
        rating = float(re.search(r'\d+', rating_text).group())
        if 1 <= rating <= 5:
            return rating
        raise ValueError("Rating out of bounds")
    except (ValueError, AttributeError):
        print(f"Unexpected response for the provided details: {rating_text}")
        return 0

def generate_combined_text_for_prediction(columns, *args):
    """
    Generates a combined text string from columns and arguments for prediction.

    Args:
        columns (list): List of column names.
        args (tuple): Values corresponding to the columns.

    Returns:
        str: Combined text string for prediction.
    """
    return ". ".join([f"{col}: {val}" for col, val in zip(columns, args)])

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
