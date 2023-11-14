import numpy as np
import pandas as pd
from constants import *

def calculate_rmse_and_mae(actual_ratings, predicted_ratings):
    differences = [actual - predicted for actual, predicted in zip(actual_ratings, predicted_ratings)]

    # RMSE (Root Mean Squared Error) Formula:
    # RMSE = sqrt(sum((actual - predicted) ** 2) / n)
    # Where:
    # - 'actual' is the array of actual values
    # - 'predicted' is the array of predicted values
    # - 'n' is the number of total predictions
    # - 'sqrt' represents the square root function
    # - '** 2' represents squaring each difference
    squared_differences = [diff ** 2 for diff in differences]
    mean_squared_difference = sum(squared_differences) / len(squared_differences)
    rmse = np.sqrt(mean_squared_difference)

    # MAE (Mean Absolute Error) Formula:
    # MAE = sum(abs(actual - predicted)) / n
    # Where:
    # - 'actual' is the array of actual values
    # - 'predicted' is the array of predicted values
    # - 'n' is the number of total predictions
    absolute_differences = [abs(diff) for diff in differences]
    mae = sum(absolute_differences) / len(absolute_differences)

    return rmse, mae

def evaluate_model_predictions_rmse_mae(data_path, num_examples, actual_ratings_column, predicted_ratings_column):
    """
    Evaluate model predictions with RMSE and MAE, and display a few prediction results.

    Parameters:
    - data_path (str): Path to the CSV file containing the model's predictions.
    - num_examples (int): Number of examples to show for debugging purposes.
    - actual_ratings_column (str): Name of the column in the CSV file containing actual ratings.
    - predicted_ratings_column (str): Name of the column in the CSV file containing predicted ratings.
    """
    # Read the data from the CSV file
    data = pd.read_csv(data_path)

    # Extract the actual and predicted ratings
    actual_ratings = data[actual_ratings_column].tolist()
    predicted_ratings = data[predicted_ratings_column].tolist()

    # Filter out invalid (None) predictions
    filtered_ratings = [(actual, predicted) for actual, predicted in zip(actual_ratings, predicted_ratings) if predicted is not None]
    
    # Check if there are valid predictions for evaluation
    if not filtered_ratings:
        print("No valid predictions available for evaluation.")
        return

    # Unpack the filtered actual and predicted ratings
    actual_filtered, predicted_filtered = zip(*filtered_ratings)

    # Calculate RMSE and MAE using the custom function
    rmse, mae = calculate_rmse_and_mae(actual_filtered, predicted_filtered)

    # Output the evaluation results
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')

    # Display the first few actual vs. predicted ratings for debugging
    print("\nFirst few actual vs predicted ratings:")
    for actual, predicted in zip(actual_filtered, predicted_filtered)[:num_examples]:
        print(f"Actual: {actual}, Predicted: {predicted}")