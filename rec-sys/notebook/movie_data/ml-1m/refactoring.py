import os
import sys
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.append('../../../')
from constants import *
from evaluation_utils import *
from path_utils import *
from ChatCompletion_OpenAI_API import *
from CF_utils import *

# File and folder paths
rec_sys_dir = get_rec_sys_directory()
DATA_DIR = os.path.join(rec_sys_dir, '../data')
DATA_PATH = os.path.join(DATA_DIR, 'ml-1m/merged_data.dat')
CF_OUTPUT_TEMPLATE = os.path.join(DATA_DIR, 'ml-1m/output/CF_fewshot_output_path_ratings_per_user_{}.dat')
CF_RERUN_TEMPLATE = os.path.join(DATA_DIR, 'ml-1m/output/rerun_CF_fewshot_output_path_ratings_per_user_{}.dat')
ZERO_SHOT_TEMPLATE = os.path.join(DATA_DIR, 'ml-1m/output/zero_shot_{}.dat')
ZERO_SHOT_RERUN_TEMPLATE = os.path.join(DATA_DIR, 'ml-1m/output/rerun_zero_shot_{}.dat')
FEW_SHOT_TEMPLATE = os.path.join(DATA_DIR, 'ml-1m/output/few_shot_{}.dat')
FEW_SHOT_RERUN_TEMPLATE = os.path.join(DATA_DIR, 'ml-1m/output/rerun_few_shot_{}.dat')

# Constants
NUM_ITERATIONS = 10
NUM_RATINGS_PER_USER = 1
NUM_MAIN_USER_RATINGS = 4
NUM_SIMILAR_USERS = 4
USER_COLUMN_NAME = 'UserID'
TITLE_COLUMN_NAME = 'Title'
ITEM_ID_COLUMN = 'MovieID'
RATING_COLUMN_NAME = 'Rating'


def load_data(data_path):
    return pd.read_csv(data_path)

def create_interaction_matrix(data):
    interaction_matrix = pd.pivot_table(data, index=USER_COLUMN_NAME, columns=ITEM_ID_COLUMN, values=RATING_COLUMN_NAME).fillna(0)
    return csr_matrix(interaction_matrix.values), interaction_matrix

def run_collaborative_filtering(data, pcc_matrix, iteration_num):
    output_path = CF_OUTPUT_TEMPLATE.format(iteration_num)
    predict_ratings_with_collaborative_filtering_and_save(
        data, pcc_matrix, save_path=output_path,
        user_column_name=USER_COLUMN_NAME, movie_column_name=TITLE_COLUMN_NAME,
        movie_id_column=ITEM_ID_COLUMN, rating_column_name=RATING_COLUMN_NAME, 
        num_ratings_per_user=NUM_RATINGS_PER_USER,
        num_main_user_ratings=NUM_MAIN_USER_RATINGS,
        num_similar_users=NUM_SIMILAR_USERS
    )
    return output_path

def rerun_failed_predictions(data, pcc_matrix, output_path, iteration_num):
    rerun_path = CF_RERUN_TEMPLATE.format(iteration_num)
    rerun_failed_CF_fewshot_predictions(
        data, pcc_matrix, save_path=output_path,
        user_column_name=USER_COLUMN_NAME, movie_column_name=TITLE_COLUMN_NAME,
        movie_id_column=ITEM_ID_COLUMN, rating_column_name=RATING_COLUMN_NAME, 
        num_ratings_per_user=NUM_RATINGS_PER_USER,
        num_main_user_ratings=NUM_MAIN_USER_RATINGS,
     ''   num_similar_users=NUM_SIMILAR_USERS,
        new_path=rerun_path, rerun_indices=get_rerun_indices(output_path)
    )
    return rerun_path

def evaluate_model(output_path):
    evaluate_model_predictions_rmse_mae(
        data_path=output_path,
        num_examples=NUM_EXAMPLES,
        actual_ratings_column='actual_rating',
        predicted_ratings_column='predicted_rating'
    )

def get_rerun_indices(output_path):
    saved_data = pd.read_csv(output_path)
    saved_data['is_rating_float'] = pd.to_numeric(saved_data['predicted_rating'], errors='coerce').notna()
    non_float_ratings = saved_data[saved_data['is_rating_float'] == False]
    return non_float_ratings.index.tolist()




def main():
    data = load_data(DATA_PATH)
    csr_matrix, interaction_matrix = create_interaction_matrix(data)
    pcc_matrix = pearson_correlation(csr_matrix)

    rmse_values = []
    mae_values = []

    for i in range(1, NUM_ITERATIONS + 1):
        print(f"\nRunning iteration {i}")
        cf_output_path = run_collaborative_filtering(data, pcc_matrix, i)
        cf_rerun_path = rerun_failed_predictions(data, pcc_matrix, cf_output_path, i)
        rmse, mae = evaluate_model(cf_rerun_path)
        if rmse is not None and mae is not None:
            rmse_values.append(rmse)
            mae_values.append(mae)

    # Calculate and print the average RMSE and MAE
    if rmse_values and mae_values:
        avg_rmse = sum(rmse_values) / len(rmse_values)
        avg_mae = sum(mae_values) / len(mae_values)
        print(f"\nAverage RMSE over {NUM_ITERATIONS} iterations: {avg_rmse:.4f}")
        print(f"Average MAE over {NUM_ITERATIONS} iterations: {avg_mae:.4f}")
    else:
        print("No valid RMSE and MAE values available for averaging.")

if __name__ == "__main__":
    main()

    

