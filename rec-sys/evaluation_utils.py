# calculate RMSE and MAE manually
def calculate_rmse_and_mae(actual_ratings, predicted_ratings):
    differences = [actual - predicted for actual,
                   predicted in zip(actual_ratings, predicted_ratings)]

    # RMSE
    squared_differences = [diff ** 2 for diff in differences]
    mean_squared_difference = sum(
        squared_differences) / len(squared_differences)
    rmse = mean_squared_difference ** 0.5

    # MAE
    absolute_differences = [abs(diff) for diff in differences]
    mae = sum(absolute_differences) / len(absolute_differences)

    return rmse, mae
