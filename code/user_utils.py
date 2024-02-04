from constants import *
import numpy as np
from sklearn.model_selection import train_test_split
import json
import gzip

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

# sequential train-test split
def sequential_train_test_split(user_data, train_ratio=0.8, time_column='Timestamp'):
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



def split_data_by_rated_items(df, user_col, test_size, given_n, random_state=RANDOM_STATE):
    """
    Splits the dataset into training and testing sets while ensuring a specified number of items 
    per user in the test set.

    Parameters:
    df (DataFrame): The dataset containing user-item interactions.
    user_col (str): The column name identifying users in the dataset.
    test_size (float): The proportion of the dataset to include in the test split.
    given_n (int): The number of items to retain for each user in the test set.
    random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
    train_df (DataFrame): The training set.
    test_df (DataFrame): The test set with 'given_n' items per user.
    """
    # Split dataset into training and testing sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[user_col])

    # Limit the number of items for each user in the test set
    test_df = test_df.groupby(user_col).apply(lambda x: x.sample(
        min(len(x), given_n), random_state=random_state))

    return train_df, test_df.reset_index(drop=True)


def split_data_by_user_percentage(df, user_col, percentages, random_state=None):
    """
    Splits the dataset into multiple subsets based on the percentage of unique users.

    Parameters:
    df (DataFrame): The dataset containing user-item interactions.
    user_col (str): The column name identifying users in the dataset.
    percentages (list of float): A list of percentages used to split the dataset.
    random_state (int): Controls the random shuffling of users.

    Returns:
    list of DataFrame: A list of DataFrames, each representing a subset of the original dataset.
    """
    unique_users = df[user_col].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_users)

    total_users = len(unique_users)
    slices = [int(p * total_users) for p in percentages]

    # Split the DataFrame into subsets based on user IDs
    sets = [df[df[user_col].isin(unique_users[slices[i]:slices[i+1]])]
            for i in range(len(slices)-1)]

    return sets


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
