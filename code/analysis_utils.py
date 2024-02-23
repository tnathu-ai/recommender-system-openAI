# Functions for performing analysis and generating reports
import pandas as pd

def compute_best_and_worst_metrics(df):
    """Compute and print the best and worst RMSE and MAE."""
    best_rmse = df['RMSE'].min()
    best_mae = df['MAE'].min()
    worst_rmse = df['RMSE'].max()
    worst_mae = df['MAE'].max()
    print(f'The best RMSE is {best_rmse} and the best MAE is {best_mae}')
    print(f'The worst RMSE is {worst_rmse} and the worst MAE is {worst_mae}')

def filter_by_method(df, method_keyword):
    """Filter DataFrame for methods containing a specific keyword."""
    return df[df['Methods'].str.contains(method_keyword)]
