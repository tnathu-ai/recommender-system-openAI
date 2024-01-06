import os
import gzip
import json
import pandas as pd
import sys

# ----------------------------------------------------------------------------------------------------------------------
# Utility functions for path management and data handling 
# ----------------------------------------------------------------------------------------------------------------------

def get_rec_sys_directory():
    """
    Get the 'rec-sys' directory, assuming this script is within the 'rec-sys' hierarchy.
    
    Returns:
        str: Absolute path to the 'rec-sys' directory.
    """
    current_dir = os.getcwd()
    rec_sys_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return rec_sys_dir

def add_path_to_sys(path):
    """
    Add a specific path to the system path if it's not already present.

    Parameters:
        path (str): The path to be added to the system path.
    """
    if path not in sys.path:
        sys.path.append(path)

def get_absolute_path(relative_path):
    """
    Convert a relative path to an absolute path.

    Parameters:
        relative_path (str): The relative path to convert.

    Returns:
        str: The absolute path.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(current_dir, relative_path)
    return absolute_path

def get_amazon_data_path(filename):
    """
    Construct the path to a data file located in the 'data/amazon-beauty' directory.

    Parameters:
        filename (str): The name of the file.

    Returns:
        str: Absolute path to the file in the 'data/amazon-beauty' directory.
    """
    relative_path = os.path.join('../../data/amazon-beauty', filename)
    return get_absolute_path(relative_path)

def get_movie_data_path(filename):
    """
    Construct the path to a data file located in the 'data/movie-ml-latest-small/' directory.

    Parameters:
        filename (str): The name of the file.

    Returns:
        str: Absolute path to the file in the 'data/movie-ml-latest-small/' directory.
    """
    relative_path = os.path.join('data/movie-ml-latest-small/', filename)
    return get_absolute_path(relative_path)

def parse_json_gz(file_path):
    """
    Parse a .json.gz file and load it into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the .json.gz file.

    Returns:
        pd.DataFrame: DataFrame containing the parsed data.
    """
    data = []

    with gzip.open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return pd.DataFrame(data)

def export_predictions_to_csv(predictions, filename):
    """
    Export a list of predictions to a CSV file.

    Parameters:
        predictions (list): A list of predictions to export.
        filename (str): The filename for the exported CSV file.
    """
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(filename, index=False)

# Read and Merge Data
def load_and_merge_data(movies_path, ratings_path, users_path):
    # Load each file
    movies = pd.read_csv(movies_path, delimiter='::', engine= 'python', header=None, names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')
    ratings = pd.read_csv(ratings_path, delimiter='::', engine= 'python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')
    users = pd.read_csv(users_path,delimiter='::', engine= 'python', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')
    # Merge datasets
    merged_data = pd.merge(pd.merge(ratings, users, on='UserID'), movies, on='MovieID')
    return merged_data