from constants import *
import numpy as np
import pandas as pd

# Set the confidence multiplier
CONFIDENCE_MULTIPLIER = 1.96  # Standard for a 95% confidence level

def calculate_rmse_and_mae(actual_ratings, predicted_ratings):
    differences = [actual - predicted for actual, predicted in zip(actual_ratings, predicted_ratings)]

    # RMSE (Root Mean Squared Error) Formula:
    squared_differences = [diff ** 2 for diff in differences]
    mean_squared_difference = sum(squared_differences) / len(squared_differences)
    rmse = np.sqrt(mean_squared_difference)

    # MAE (Mean Absolute Error) Formula:
    absolute_differences = [abs(diff) for diff in differences]
    mae = sum(absolute_differences) / len(absolute_differences)

    return rmse, mae

def calculate_confidence_interval(data, confidence_multiplier):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    h = std_err * confidence_multiplier
    return mean - h, mean + h

def evaluate_model_predictions_rmse_mae(data_path, num_examples, actual_ratings_column, predicted_ratings_column):
    """
    Evaluate model predictions with RMSE and MAE, their confidence intervals, and margins of error.
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
        return None, None

    # Unpack the filtered actual and predicted ratings
    actual_filtered, predicted_filtered = zip(*filtered_ratings)

    # Calculate RMSE and MAE
    rmse, mae = calculate_rmse_and_mae(actual_filtered, predicted_filtered)

    # Confidence intervals for RMSE and MAE
    rmse_conf_interval = calculate_confidence_interval([rmse] * len(filtered_ratings), CONFIDENCE_MULTIPLIER)
    mae_conf_interval = calculate_confidence_interval([mae] * len(filtered_ratings), CONFIDENCE_MULTIPLIER)

    # Margins of error
    rmse_error = (rmse_conf_interval[1] - rmse_conf_interval[0]) / 2
    mae_error = (mae_conf_interval[1] - mae_conf_interval[0]) / 2

    # Formatting the output to four decimal points
    print(f'RMSE: {rmse:.4f} (95% CI: ({rmse_conf_interval[0]:.4f}, {rmse_conf_interval[1]:.4f})) ± {rmse_error:.4f}')
    print(f'MAE: {mae:.4f} (95% CI: ({mae_conf_interval[0]:.4f}, {mae_conf_interval[1]:.4f})) ± {mae_error:.4f}')

    # Display the first few actual vs. predicted ratings for debugging
    print("\nFirst few actual vs predicted ratings:")
    for actual, predicted in list(zip(actual_filtered, predicted_filtered))[:num_examples]:
        print(f"Actual: {actual}, Predicted: {predicted:.4f}")
        
    return rmse, mae

def calculate_average_rmse_mae_over_runs(runs_data_paths, actual_ratings_column, predicted_ratings_column):
    rmses = []
    maes = []

    for data_path in runs_data_paths:
        data = pd.read_csv(data_path)
        rmse, mae = evaluate_model_predictions_rmse_mae(data, actual_ratings_column, predicted_ratings_column, print_results=False)
        if rmse is not None and mae is not None:
            rmses.append(rmse)
            maes.append(mae)

    average_rmse = np.mean(rmses) if rmses else None
    average_mae = np.mean(maes) if maes else None
    print(f"Average RMSE over {len(rmses)} runs: {average_rmse:.4f}")
    print(f"Average MAE over {len(maes)} runs: {average_mae:.4f}")

    return average_rmse, average_mae
