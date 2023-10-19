import os
import pandas as pd
import numpy as np
import openai
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt
import sys
import re

# Constants and Configuration
CONFIG = {
    'OPENAI_API_KEY': 'YOUR_API_KEY',
    'DATA_PATH': '../../data/amazon-beauty/large_merged_data.csv',
    'PAUSE_EVERY_N_USERS': 10,
    'SLEEP_TIME': 60,
    # Add other constants as needed
}

# Utility Functions


def load_data(path):
    """
    Load data from the provided path.
    """
    # Load and return data
    pass


def calculate_rmse_and_mae(actual_ratings, predicted_ratings):
    """
    Calculate RMSE and MAE.
    """
    # Calculate and return RMSE and MAE
    pass


def predict_rating_zero_shot(title, review):
    """
    Predict ratings using zero-shot approach.
    """
    # Prediction logic
    pass


def predict_rating_few_shot(title, review, rating_history_str):
    """
    Predict ratings using few-shot approach.
    """
    # Prediction logic
    pass

# Main Execution


# Utility Functions

def export_predictions_to_csv(predictions, filename):
    """
    Export the predicted ratings to a CSV file.
    """
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(filename, index=False)

# Main Execution


def main():
    # Load data
    data = load_data(CONFIG['DATA_PATH'])

    # Zero-shot prediction
    zero_shot_predictions = predict_rating_zero_shot(data)
    export_predictions_to_csv(zero_shot_predictions,
                              'zero_shot_predictions.csv')

    # Few-shot prediction
    few_shot_predictions = predict_rating_few_shot(data)
    export_predictions_to_csv(few_shot_predictions, 'few_shot_predictions.csv')

    # Evaluation
    rmse, mae = calculate_rmse_and_mae(data['rating'], few_shot_predictions)
    print(f"RMSE: {rmse}, MAE: {mae}")


if __name__ == "__main__":
    main()
