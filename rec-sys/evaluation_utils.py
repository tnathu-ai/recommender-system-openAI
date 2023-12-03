import numpy as np
import pandas as pd
from constants import *
import numpy as np
import pandas as pd
from sklearn.utils import resample

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


def calculate_standard_error(values):
    """ Calculate the standard error of the mean (SEM) for a list of values. """
    n = len(values)
    if n <= 1:
        return 0
    return np.std(values, ddof=1) / np.sqrt(n)

def bootstrap_rmse_mae_combined(actual, predicted, n_iterations=BOOSTRAP_RESAMPLING_ITERATIONS, confidence_level=CONFIDENCE_LEVEL, confidence_multiplier=CONFIDENCE_MULTIPLIER):
    bootstrapped_rmse = []
    bootstrapped_mae = []

    for _ in range(n_iterations):
        # Bootstrap resample with replacement
        sample_actual, sample_predicted = resample(actual, predicted)
        sample_rmse, sample_mae = calculate_rmse_and_mae(sample_actual, sample_predicted)
        bootstrapped_rmse.append(sample_rmse)
        bootstrapped_mae.append(sample_mae)

    # Confidence intervals
    lower_percentile = ((1 - confidence_level) / 2) * 100
    upper_percentile = (confidence_level + (1 - confidence_level) / 2) * 100
    rmse_conf_interval = np.percentile(bootstrapped_rmse, [lower_percentile, upper_percentile])
    mae_conf_interval = np.percentile(bootstrapped_mae, [lower_percentile, upper_percentile])

    # Margins of error
    rmse_error = calculate_standard_error(bootstrapped_rmse) * confidence_multiplier
    mae_error = calculate_standard_error(bootstrapped_mae) * confidence_multiplier

    return (np.mean(bootstrapped_rmse), rmse_conf_interval, rmse_error), (np.mean(bootstrapped_mae), mae_conf_interval, mae_error)

def evaluate_model_predictions_rmse_mae(data_path, num_examples, actual_ratings_column, predicted_ratings_column):
    """
    Evaluate model predictions with RMSE and MAE, their 95% confidence intervals, and margins of error.
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

    # Calculate RMSE and MAE with confidence intervals and margins of error
    (rmse, rmse_conf_interval, rmse_error), (mae, mae_conf_interval, mae_error) = bootstrap_rmse_mae_combined(actual_filtered, predicted_filtered)

    # Formatting the output to four decimal points
    print(f'RMSE: {rmse:.4f} (95% CI: ({rmse_conf_interval[0]:.4f}, {rmse_conf_interval[1]:.4f})) ± {rmse_error:.4f}')
    print(f'MAE: {mae:.4f} (95% CI: ({mae_conf_interval[0]:.4f}, {mae_conf_interval[1]:.4f})) ± {mae_error:.4f}')

    # Display the first few actual vs. predicted ratings for debugging
    print("\nFirst few actual vs predicted ratings:")
    for actual, predicted in list(zip(actual_filtered, predicted_filtered))[:num_examples]:
        print(f"Actual: {actual}, Predicted: {predicted:.4f}")

