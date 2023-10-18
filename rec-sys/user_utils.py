from constants import *
import numpy as np
from sklearn.model_selection import train_test_split
import json
import gzip


def split_data_by_rated_items(df, user_col, test_size, given_n, random_state=RANDOM_STATE):
    """
    Splits the data into a training set and a test set. For each user in the test set, 
    it keeps only a given number of rated items.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[user_col])
    test_df = test_df.groupby(user_col).apply(lambda x: x.sample(
        min(len(x), given_n), random_state=random_state))
    return train_df, test_df.reset_index(drop=True)


def split_data_by_user_percentage(df, user_col, percentages, random_state=None):
    """
    Splits the data into different sets based on percentages of unique users.
    """
    unique_users = df[user_col].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_users)

    total_users = len(unique_users)
    slices = [int(p * total_users) for p in percentages]

    # Split the DataFrame into the different sets based on the user IDs
    sets = [df[df[user_col].isin(unique_users[slices[i]:slices[i+1]])]
            for i in range(len(slices)-1)]

    return sets


def all_but_one(df, user_col, random_state=None):
    """
    For each user, select one rating and split it into a separate DataFrame.
    """
    test_df = df.groupby(user_col).sample(n=1, random_state=random_state)
    train_df = df.drop(test_df.index)
    return train_df, test_df
