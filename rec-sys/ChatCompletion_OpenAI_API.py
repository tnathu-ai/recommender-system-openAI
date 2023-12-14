import numpy as np
import openai
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import time
from constants import *
from evaluation_utils import *
from utils import *
from CF_utils import *
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import openai
from openai.error import APIError, APIConnectionError, RateLimitError


@retry_decorator
def predict_rating_combined_ChatCompletion(combined_text, 
                                           model=GPT_MODEL_NAME, 
                                           temperature=TEMPERATURE, 
                                           approach="zero-shot", 
                                           rating_history=None, 
                                           similar_users_ratings=None, 
                                           seed=RANDOM_STATE, 
                                           system_content=AMAZON_CONTENT_SYSTEM):
    # Validation
    if approach == "few-shot" and rating_history is None:
        raise ValueError("Rating history is required for the few-shot approach.")
    if approach == "CF" and similar_users_ratings is None:
        raise ValueError("Similar users' ratings are required for the collaborative filtering approach.")
    if not system_content:
        raise ValueError("System content is required.")

    # Check and reduce length of combined_text
    combined_text = check_and_reduce_length(combined_text, MAX_TOKENS_CHAT_GPT // 3, TOKENIZER)
    prompt = f"How will user rate this {combined_text}? (1 being lowest and 5 being highest) Attention! Just give me back the exact number as a result, and you don't need a lot of text."

    # Construct the prompt based on the approach
    if approach == "few-shot":
        rating_history = check_and_reduce_length(rating_history, MAX_TOKENS_CHAT_GPT // 3, TOKENIZER)
        prompt += f"\n\nHere is user rating history:\n{rating_history}"

    elif approach == "CF":
        similar_users_ratings = check_and_reduce_length(similar_users_ratings, MAX_TOKENS_CHAT_GPT // 3, TOKENIZER)
        prompt += f"\n\nHere are the rating history from users who are similar to this user:\n{similar_users_ratings}"

    # Adding end of the prompt
    prompt += "\n\nBased on the above information, please predict user's rating for the product: (1 being lowest and 5 being highest, The output should be like: (x stars, xx%), do not explain the reason.)"

    print(f"Constructed Prompt for {approach} approach:\n")
    print(f'The prompt:\n**********\n{prompt}\n**********\n')

    try:
        # Create the API call
        response = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            max_tokens=MAX_TOKENS_CHAT_GPT,
            seed=seed,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract the system fingerprint and print it
        system_fingerprint = response.get('system_fingerprint')
        print(f"\n\nSystem Fingerprint: {system_fingerprint}")
        # Extract and return the rating
        rating_text = response.choices[0].message['content'].strip()
        print(f'\nAPI call response: "{rating_text}"')
        extracted_rating = extract_numeric_rating(rating_text)
        print(f'Extracted rating: {extracted_rating}')
        return extracted_rating  # Ensure this is a float
    
    except APIError as api_err:
        print(f"API Error occurred: {api_err}")
        return None, str(api_err)
    except RateLimitError as rate_err:
        print(f"Rate Limit Error occurred: {rate_err}")
        return None, str(rate_err)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None, str(e)


def predict_ratings_zero_shot_and_save(data,
                                       columns_for_prediction=['title'],
                                       user_column_name='reviewerID',
                                       title_column_name='title',
                                       asin_column_name='asin',
                                       rating_column_name='rating',
                                       pause_every_n_users=PAUSE_EVERY_N_USERS,
                                       sleep_time=SLEEP_TIME,
                                       save_path='zero_shot_predictions.csv',
                                       seed=RANDOM_STATE):
    """
    Predicts a single random rating per user using a zero-shot approach and saves the predictions to a CSV file.

    Parameters:
    - data (DataFrame): Dataset containing user ratings.
    - columns_for_prediction (list of str): Columns to use for prediction.
    - user_column_name (str): Column name for user IDs.
    - title_column_name (str): Column name for item titles.
    - asin_column_name (str): Column name for item IDs.
    - rating_column_name (str): Column name for actual ratings.
    - pause_every_n_users (int): Number of users to process before pausing.
    - sleep_time (int): Sleep time in seconds during pause.
    - save_path (str): Path to save the predictions CSV file.
    - seed (int): Seed for random number generation.

    Returns:
    - DataFrame: DataFrame containing prediction results.
    """

    results = []
    random.seed(seed)

    # Group data by user and filter users with at least 5 records
    grouped_data = data.groupby(user_column_name).filter(lambda x: len(x) >= 5)
    unique_users = grouped_data[user_column_name].unique()

    for i, user_id in enumerate(unique_users):
        user_data = grouped_data[grouped_data[user_column_name] == user_id]
        # Select a random record for each user
        random_row = user_data.sample(n=1, random_state=seed).iloc[0]

        # Generate combined text for prediction using specified columns
        combined_text = ' | '.join([f"{col}: {random_row[col]}" for col in columns_for_prediction])

        # Predict rating using zero-shot approach
        predicted_rating = predict_rating_combined_ChatCompletion(combined_text, approach="zero-shot")
        item_id = random_row[asin_column_name]
        actual_rating = random_row[rating_column_name]
        title = random_row[title_column_name]

        results.append([user_id, item_id, title, actual_rating, predicted_rating])

        # Print progress and pause if necessary
        if (i + 1) % pause_every_n_users == 0:
            print(f"Processed {i + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating'])
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    return results_df



def rerun_failed_zero_shot_predictions(data, 
                                       save_path, 
                                       rerun_save_path, 
                                       columns_for_prediction, 
                                       pause_every_n_users, 
                                       sleep_time,
                                       user_column_name='user_id', 
                                       asin_column_name='item_id'):
    # Identify failed predictions
    data['is_rating_float'] = pd.to_numeric(data['predicted_rating'], errors='coerce').notna()
    failed_indices = data[~data['is_rating_float']].index

    if len(failed_indices) > 0:
        print(f"Re-running predictions for {len(failed_indices)} failed cases.")
        failed_data = data.loc[failed_indices]
        updated_data = predict_ratings_zero_shot_and_save(
            failed_data, 
            columns_for_prediction=columns_for_prediction, 
            save_path=rerun_save_path, 
            pause_every_n_users=pause_every_n_users, 
            sleep_time=sleep_time, user_column_name=user_column_name, asin_column_name=asin_column_name)
        data.loc[failed_indices, 'predicted_rating'] = updated_data['predicted_rating']

    data.to_csv(save_path, index=False)
    return data


def predict_ratings_few_shot_and_save(data, 
                                      columns_for_training, 
                                      columns_for_prediction, 
                                      user_column_name='reviewerID', 
                                      title_column_name='title', 
                                      asin_column_name='asin', 
                                      rating_column_name='rating',
                                      obs_per_user=None, 
                                      pause_every_n_users=PAUSE_EVERY_N_USERS, 
                                      sleep_time=SLEEP_TIME, 
                                      save_path='few_shot_predictions.csv'):
    results = []
    users = data[user_column_name].unique()

    for idx, user_id in enumerate(users):
        user_data = data[data[user_column_name] == user_id]

        if len(user_data) < 5:
            continue

        user_data = user_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        for test_idx, test_row in user_data.iterrows():
            # Skip the current item being predicted
            train_data = user_data[user_data[asin_column_name] != test_row[asin_column_name]]

            # Select 4 distinct previous ratings
            if len(train_data) >= 4:
                train_data = train_data.head(4)
            else:
                continue  # Skip if there are not enough historical ratings

            prediction_data = {col: test_row[col] for col in columns_for_prediction if col != rating_column_name}
            combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data.values())

            rating_history_str = '\n'.join([
                '* ' + ' | '.join(f"{col}: {row[col]}" for col in columns_for_training) + f" - Rating: {row[rating_column_name]} stars"
                for _, row in train_data.iterrows()
            ])

            predicted_rating = predict_rating_combined_ChatCompletion(combined_text, rating_history=rating_history_str, approach="few-shot")

            item_id = test_row[asin_column_name]
            actual_rating = test_row[rating_column_name]
            title = test_row[title_column_name]

            results.append([user_id, item_id, title, actual_rating, predicted_rating])

            if obs_per_user and len(results) >= obs_per_user:
                break

        if (idx + 1) % pause_every_n_users == 0:
            print(f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating'])
    results_df.to_csv(save_path, index=False)
    print("Predictions saved to", save_path)



def predict_ratings_with_collaborative_filtering_and_save(data, 
                                                          pcc_matrix, 
                                                          user_column_name='reviewerID', 
                                                          movie_column_name='title', 
                                                          movie_id_column='asin',
                                                          rating_column_name='rating', 
                                                          num_ratings_per_user=1, 
                                                          num_similar_users=4, 
                                                          save_path='cf_predictions.csv', 
                                                          seed=RANDOM_STATE):
    """
    Predicts movie ratings using a user-based collaborative filtering approach 
    and saves the predictions to a CSV file.

    Parameters:
    - data (DataFrame): Dataset containing user ratings.
    - pcc_matrix (ndarray): Pearson Correlation Coefficient matrix.
    - user_column_name (str): Column name for user IDs.
    - movie_column_name (str): Column name for movie titles.
    - movie_id_column (str): Column name for movie IDs.
    - rating_column_name (str): Column name for ratings.
    - num_ratings_per_user (int): Number of historical ratings to consider per similar user.
    - num_similar_users (int): Number of similar users to consider for prediction.
    - save_path (str): Path to save the predictions CSV file.
    - seed (int): Seed for random number generation.

    Returns:
    - DataFrame: DataFrame containing prediction results.
    """

    results = []
    unique_users = data[user_column_name].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}

    random.seed(seed)

    for user_id in unique_users:
        user_idx = user_id_to_index[user_id]

        # Find the top similar users based on Pearson Correlation Coefficient
        similar_users_idx = np.argsort(-pcc_matrix[user_idx])[:num_similar_users + 1]
        similar_users_idx = similar_users_idx[similar_users_idx != user_idx][:num_similar_users]

        print(f"User {user_id} - Similar Users: {[unique_users[idx] for idx in similar_users_idx]}")

        # Collect historical ratings from similar users
        similar_users_ratings = ""
        for idx in similar_users_idx:
            similar_user_id = unique_users[idx]
            similar_user_data = data[data[user_column_name] == similar_user_id]
            historical_ratings = similar_user_data.head(num_ratings_per_user)
            for _, row in historical_ratings.iterrows():
                rating_info = f"* Title: {row[movie_column_name]}, Rating: {row[rating_column_name]} stars"
                print(f"    Similar User {similar_user_id} - Historical Rating: {rating_info}")
                similar_users_ratings += rating_info + "\n"

        # Select a random movie from the user's data for prediction
        user_data = data[data[user_column_name] == user_id]
        random_movie_row = user_data.sample(n=1, random_state=seed).iloc[0]
        random_movie_title = random_movie_row[movie_column_name]
        random_movie_id = random_movie_row[movie_id_column]
        actual_rating = random_movie_row[rating_column_name]

        # Predict rating using collaborative filtering
        combined_text = f"Title: {random_movie_title}"
        predicted_rating = predict_rating_combined_ChatCompletion(
            combined_text, 
            approach="CF", 
            similar_users_ratings=similar_users_ratings
        )

        # Store prediction results
        results.append([user_id, random_movie_id, random_movie_title, actual_rating, predicted_rating])

    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating'])
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    return results_df


def predict_ratings_with_collaborative_filtering_and_save(data, 
                                                          pcc_matrix, 
                                                          user_column_name='UserID', 
                                                          movie_column_name='Title', 
                                                          movie_id_column='MovieID',
                                                          rating_column_name='Rating', 
                                                          num_ratings_per_user=1, 
                                                          num_similar_users=4, 
                                                          save_path='cf_predictions.csv', 
                                                          seed=RANDOM_STATE):
    """
    Predicts movie ratings using a user-based collaborative filtering approach 
    and saves the predictions to a CSV file.

    Parameters:
    - data (DataFrame): The dataset containing user ratings.
    - pcc_matrix (ndarray): Pearson Correlation Coefficient matrix.
    - user_column_name (str): Name of the column representing user IDs.
    - movie_column_name (str): Name of the column representing movie titles.
    - movie_id_column (str): Name of the column representing movie IDs.
    - rating_column_name (str): Name of the column representing ratings.
    - num_ratings_per_user (int): Number of historical ratings to consider per similar user.
    - num_similar_users (int): Number of similar users to consider for prediction.
    - save_path (str): Path to save the predictions CSV file.
    - seed (int): Seed for random number generation.

    Returns:
    - DataFrame: A DataFrame containing the prediction results.
    """

    results = []
    users = data[user_column_name].unique()
    random.seed(seed)

    for user_id in users:
        user_idx = user_id - 1  # Adjust for zero-based indexing

        # Identify top similar users
        similar_users_idx = np.argsort(-pcc_matrix[user_idx])[:num_similar_users + 1]
        similar_users_idx = similar_users_idx[similar_users_idx != user_idx][:num_similar_users]

        # Collect historical ratings from similar users
        similar_users_ratings = ""
        for idx in similar_users_idx:
            similar_user_id = idx + 1
            similar_user_data = data[data[user_column_name] == similar_user_id]
            historical_ratings = similar_user_data.head(num_ratings_per_user)
            for _, row in historical_ratings.iterrows():
                rating_info = f"* title: {row[movie_column_name]} - Rating: {row[rating_column_name]} stars"
                similar_users_ratings += rating_info + "\n"

        # Select a random movie from the user's data for prediction
        user_data = data[data[user_column_name] == user_id]
        random_movie_row = user_data.sample(n=1, random_state=seed).iloc[0]
        random_movie_title = random_movie_row[movie_column_name]
        random_movie_id = random_movie_row[movie_id_column]
        actual_rating = random_movie_row[rating_column_name]

        # Predict rating using collaborative filtering
        combined_text = f"title: {random_movie_title}"
        predicted_rating = predict_rating_combined_ChatCompletion(
            combined_text, 
            approach="CF", 
            similar_users_ratings=similar_users_ratings
        )

        # Store prediction results
        results.append([user_id, random_movie_id, random_movie_title, actual_rating, predicted_rating])

    # Save predictions to CSV file
    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating'])
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    return results_df
