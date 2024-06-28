from constants import *
import numpy as np
from sklearn.model_selection import train_test_split
import json
import gzip
import pandas as pd
from sklearn.utils import resample

# sequential train-test split
def sequential_train_test_split(user_data, train_ratio=TRAIN_RATIO, time_column='Timestamp'):
    """
    Sequentially split user data into training and test sets based on timestamps.

    Args:
    - user_data (DataFrame): Data for the specific user, containing a timestamp column.
    - train_ratio (float): Proportion of the dataset to include in the train split.
    - time_column (str): Name of the column containing the timestamp.

    Returns:
    - DataFrame: Training set for the user.
    - DataFrame: Test set for the user.
    """
    # Sort the data by timestamp
    user_data_sorted = user_data.sort_values(by=time_column)

    # Calculate the index at which to split the data
    split_index = int(len(user_data_sorted) * train_ratio)

    # Split the data into training and test sets
    remaining_data = user_data_sorted.iloc[:split_index]
    test_set = user_data_sorted.iloc[split_index:]

    return test_set, remaining_data

# Random train-test split
def select_test_set_for_user(user_data, num_tests=TEST_OBSERVATION_PER_USER, seed=RANDOM_STATE):
    """
    Select a consistent test set for a given user.

    Args:
    - user_data (DataFrame): Data for the specific user.
    - num_tests (int): Number of test samples to select.
    - seed (int): Random state seed for consistency.

    Returns:
    - DataFrame: Test set for the user.
    - DataFrame: Remaining data after removing the test set.
    """
    test_set = user_data.sample(n=num_tests, random_state=seed)
    remaining_data = user_data.drop(test_set.index)
    return test_set, remaining_data

# Random Popularity Split
def popularity_based_random_split(data, 
                                  item_column='asin', 
                                  review_column='reviewText', 
                                  rating_column='rating', 
                                  test_ratio=TEST_RATIO, 
                                  seed=RANDOM_STATE, 
                                  test_set_type='both'):
    """
    Randomly split user data into training and test sets based on item popularity,
    allowing selection of either popular, unpopular, or both types of items in the test set.

    Args:
    - data (DataFrame): Dataset containing user data, item identifiers, and timestamps.
    - item_column (str): Name of the column containing the item identifier.
    - review_column (str): Name of the column containing review content to ensure valid entries.
    - rating_column (str): Name of the column containing ratings.
    - test_ratio (float): Proportion of the dataset to include in the test split.
    - seed (int): Seed for the random number generator for reproducibility.
    - test_set_type (str): Type of test set to return ('popular', 'unpopular', 'both').

    Returns:
    - DataFrame: Training set.
    - DataFrame: Test set.
    """
    # Preprocessing
    data = data.dropna(subset=[item_column, review_column, rating_column])

    # Calculate popularity
    item_counts = data[item_column].value_counts()
    average_ratings = data.groupby(item_column)[rating_column].mean()
    popularity_score = (item_counts * 0.5) + (average_ratings * 0.5 * item_counts.max() / average_ratings.max())
    popularity_score = popularity_score.sort_values(ascending=False)

    # Identify top 20% popular items
    top_20_percent_cutoff = int(len(popularity_score) * 0.2)
    popular_items = popularity_score.head(top_20_percent_cutoff).index

    # Split entire dataset randomly first
    train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=seed)

    # Filter test set based on type
    if test_set_type == 'popular':
        test_set = test_data[test_data[item_column].isin(popular_items)]
    elif test_set_type == 'unpopular':
        test_set = test_data[~test_data[item_column].isin(popular_items)]
    else:
        popular_test_set = test_data[test_data[item_column].isin(popular_items)]
        unpopular_test_set = test_data[~test_data[item_column].isin(popular_items)]
        test_set = pd.concat([popular_test_set, unpopular_test_set])

    return train_data, test_set



# Sequential Popularity Split
def popularity_based_sequential_split(data, 
                                      item_column='asin', 
                                      review_column='reviewText', 
                                      rating_column='rating', 
                                      time_column='Timestamp', 
                                      test_ratio=TEST_RATIO, 
                                      test_set_type='both'):
    """
    Split user data into training and test sets based on item popularity,
    allowing selection of either popular, unpopular, or both types of items in the test set,
    preserving the temporal sequence within the test data.

    Args:
    - data (DataFrame): Dataset containing user data, item identifiers, and timestamps.
    - item_column (str): Name of the column containing the item identifier.
    - review_column (str): Name of the column containing review content to ensure valid entries.
    - rating_column (str): Name of the column containing ratings.
    - time_column (str): Name of the column containing the timestamp.
    - test_ratio (float): Proportion of the dataset to include in the test split.
    - test_set_type (str): Type of test set to return ('popular', 'unpopular', 'both').

    Returns:
    - DataFrame: Training set for the user.
    - DataFrame: Test set for the user.
    """
    # Preprocessing
    data = data.dropna(subset=[item_column, review_column, rating_column])

    # Calculate popularity
    item_counts = data[item_column].value_counts()
    average_ratings = data.groupby(item_column)[rating_column].mean()
    popularity_score = (item_counts * 0.5) + (average_ratings * 0.5 * item_counts.max() / average_ratings.max())
    popularity_score = popularity_score.sort_values(ascending=False)

    # Identify top 20% popular items
    top_20_percent_cutoff = int(len(popularity_score) * 0.2)
    popular_items = popularity_score.head(top_20_percent_cutoff).index

    # Sequentially split the entire dataset first
    train_data, test_data = data[:int(len(data) * (1 - test_ratio))], data[int(len(data) * (1 - test_ratio)):]

    # Filter test set based on type
    if test_set_type == 'popular':
        test_set = test_data[test_data[item_column].isin(popular_items)]
    elif test_set_type == 'unpopular':
        test_set = test_data[~test_data[item_column].isin(popular_items)]
    else:
        popular_test_set = test_data[test_data[item_column].isin(popular_items)]
        unpopular_test_set = test_data[~test_data[item_column].isin(popular_items)]
        test_set = pd.concat([popular_test_set, unpopular_test_set])

    return train_data, test_set



# --------------------------------------------------



def all_but_one(df, user_col, random_state=None):
    """
    For each user, selects one rating and creates a separate DataFrame with these ratings.

    Parameters:
    df (DataFrame): The dataset containing user-item interactions.
    user_col (str): The column name identifying users in the dataset.
    random_state (int): Controls the random selection of ratings for each user.

    Returns:
    train_df (DataFrame): The training set.
    test_df (DataFrame): The test set with one rating per user.
    """
    # Select one rating per user for the test set
    test_df = df.groupby(user_col).sample(n=1, random_state=random_state)
    
    # Create the training set by dropping the selected test set ratings
    train_df = df.drop(test_df.index)

    return train_df, test_df


# Filter Users with ≥ 5 Ratings
def filter_users(data):
    user_rating_counts = data['UserID'].value_counts()
    valid_users = user_rating_counts[user_rating_counts >= 5].index.tolist()
    return data[data['UserID'].isin(valid_users)]
