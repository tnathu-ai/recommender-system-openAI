import os
import gzip
import json
import pandas as pd


def get_absolute_path(relative_path):
    """Get the absolute path given a relative path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(current_dir, relative_path)
    return absolute_path


def get_amazon_data_path(filename):
    """Get the path to a data file in the 'data/amazon-beauty' directory."""
    relative_path = os.path.join('../../data/amazon-beauty', filename)
    return get_absolute_path(relative_path)


def get_movie_data_path(filename):
    """Get the path to a data file in the 'data/movie-ml-latest-small/' directory."""
    relative_path = os.path.join(
        'data/movie-ml-latest-small/', filename)
    return get_absolute_path(relative_path)


def parse_json_gz(file_path):
    """
    Parse a .json.gz file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the .json.gz file.

    Returns:
    - DataFrame: A pandas DataFrame containing the parsed data.
    """
    data = []

    with gzip.open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return pd.DataFrame(data)


def export_predictions_to_csv(predictions, filename):
    """
    Export the predicted ratings to a CSV file.
    """
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(filename, index=False)
