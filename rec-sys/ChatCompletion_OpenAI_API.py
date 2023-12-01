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
    """
    Predicts product ratings using different approaches with the GPT model.
    """
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
    # meaningful print for prompt
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
    except Exception as e:
        print(f"Error occurred during API call: {e}")
        return None, str(e)  # Include the error message with the response

    # Extract the system fingerprint and print it
    system_fingerprint = response.get('system_fingerprint')
    print(f"\n\nSystem Fingerprint: {system_fingerprint}")

    # Extract and return the rating
    rating_text = response.choices[0].message['content'].strip()
    print(f'\nAPI call response: "{rating_text}"')
    extracted_rating = extract_numeric_rating(rating_text)
    return extracted_rating



def predict_ratings_zero_shot_and_save(data,
                                       columns_for_prediction=['title'],
                                       user_column_name='reviewerID', # Parameter for user column name
                                       title_column_name='title',     # Parameter for title column name
                                       asin_column_name=None,         # Parameter for ASIN column name, optional
                                       pause_every_n_users=PAUSE_EVERY_N_USERS,
                                       sleep_time=SLEEP_TIME,
                                       save_path='../../data/amazon-beauty/reviewText_large_predictions_zero_shot.csv'):
    predicted_ratings = []

    # Group data by user and filter users with at least 5 records
    grouped_data = data.groupby(user_column_name).filter(lambda x: len(x) >= 5)

    for idx, row in grouped_data.iterrows():
        # Generate combined text for prediction with column names
        combined_text = ' | '.join([f"{col}: {row[col]}" for col in columns_for_prediction])

        predicted_rating = predict_rating_combined_ChatCompletion(combined_text, approach="zero-shot")

        product_title = row.get(title_column_name, "Unknown Title")
        product_details = f"{product_title}"
        
        if asin_column_name and asin_column_name in row:
            product_code = row[asin_column_name]
            product_details += f" (Code: {product_code})"

        print(f"Processing item {idx + 1}/{len(grouped_data)}\n")
        print(f"Details: {product_details}")
        print(f"\nPredicted Rating: {predicted_rating} stars")
        print(f"\n------------------------------------\n")

        predicted_ratings.append(predicted_rating)

        if (idx + 1) % pause_every_n_users == 0:
            print(f"Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    grouped_data['zero_shot_predicted_rating'] = predicted_ratings
    grouped_data.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

def predict_ratings_few_shot_and_save(data, 
                                      columns_for_training, 
                                      columns_for_prediction, 
                                      user_column_name='reviewerID', 
                                      title_column_name='title', 
                                      asin_column_name='asin', 
                                      obs_per_user=None, 
                                      pause_every_n_users=PAUSE_EVERY_N_USERS, 
                                      sleep_time=SLEEP_TIME, 
                                      save_path='../../data/amazon-beauty/reviewText_small_predictions_few_shot.csv'):
    predicted_ratings = []
    actual_ratings = []
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

            prediction_data = {col: test_row[col] for col in columns_for_prediction if col != 'rating'}
            combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data.values())

            rating_history_str = '\n'.join([
                '* ' + ' | '.join(f"{col}: {row[col]}" for col in columns_for_training) + f" - Rating: {row['rating']} stars"
                for _, row in train_data.iterrows()
            ])

            predicted_rating = predict_rating_combined_ChatCompletion(combined_text, rating_history=rating_history_str, approach="few-shot")

            product_title = test_row.get(title_column_name, "Unknown Title")
            product_details = f"{product_title}"
            if asin_column_name in test_row:
                product_code = test_row[asin_column_name]
                product_details += f" (Code: {product_code})"

            print(f"Processing user {idx + 1}/{len(users)}, item {test_idx + 1}/{len(user_data)}")
            print(f"User {user_id}:")
            print(f"Rating History for Prediction:\n{rating_history_str}")
            print(f"Predicted Item: {product_details}")
            print(f"Predicted Rating: {predicted_rating} stars\n")
            print("----------------\n")

            predicted_ratings.append(predicted_rating)
            actual_ratings.append(test_row['rating'])

            if obs_per_user and len(predicted_ratings) >= obs_per_user:
                break  # Break if the observation per user limit is reached

        if (idx + 1) % pause_every_n_users == 0:
            print(f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    predicted_ratings_df = pd.DataFrame({'few_shot_predicted_rating': predicted_ratings, 'actual_rating': actual_ratings})
    predicted_ratings_df.to_csv(save_path, index=False)


# User based Collaborative Filtering
def predict_ratings_with_collaborative_filtering_and_save(data, interaction_matrix, user_mapper, item_mapper, user_inv_mapper, model_knn,
                                                          columns_for_training, columns_for_prediction,
                                                          title_column_name='title', user_column_name='reviewerID',
                                                          asin_column_name='asin',
                                                          obs_per_user=None, pause_every_n_users=5, sleep_time=5,
                                                          save_path='similar_users_predictions.csv'):
    all_similar_users_ratings = get_all_similar_users_ratings(data, user_mapper, user_inv_mapper, model_knn, interaction_matrix, title_column_name)
    predicted_ratings = []
    actual_ratings = []

    for idx, user_id in enumerate(data[user_column_name].unique()):
        user_data = data[data[user_column_name] == user_id].sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"\n-------------------\nUser {user_id} ({idx + 1}/{len(data[user_column_name].unique())}):")
        if len(user_data) < 5:
            print(f"Insufficient data points for user {user_id}, skipping...")
            continue

        for test_idx, test_row in user_data.iterrows():
            train_data = user_data[user_data[asin_column_name] != test_row[asin_column_name]]
            n_train_items = min(4, len(train_data))
            if n_train_items == 0:
                print(f"No training data for user {user_id}, item {test_row[asin_column_name]}, skipping...")
                continue

            similar_users_ratings_str = format_similar_users_ratings(all_similar_users_ratings.get(user_id, []))
            print(f"\nSimilar users' ratings:\n{similar_users_ratings_str}")
            if not similar_users_ratings_str:
                print(f"No similar users' ratings found for user {user_id}, skipping prediction.")
                continue

            prediction_data = {col: test_row[col] for col in columns_for_prediction if col != 'rating'}
            print(f"\nPrediction data: {prediction_data}")
            combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data.values())
            print(f"\nCombined text: {combined_text}")

            try:
                predicted_rating = predict_rating_combined_ChatCompletion(
                    combined_text,
                    rating_history=similar_users_ratings_str,
                    approach="CF"
                )
                if predicted_rating is None:
                    print("\nPrediction failed, skipping...")
                    continue
            except Exception as e:
                print(f"\nError during prediction: {e}, skipping...")
                continue

            product_title = test_row.get(title_column_name, "Unknown Title")
            product_details = f"{product_title} (Code: {test_row[asin_column_name]})" if asin_column_name in test_row else product_title

            print(f"Predicted Rating - {predicted_rating} stars for '{product_details}'")
            predicted_ratings.append(predicted_rating)
            actual_ratings.append(test_row['rating'])

            if obs_per_user and len(predicted_ratings) >= obs_per_user:
                break

        if (idx + 1) % pause_every_n_users == 0:
            print(f"\nProcessed {idx + 1} users, pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    predicted_ratings_df = pd.DataFrame({'predicted_rating': predicted_ratings, 'actual_rating': actual_ratings})
    predicted_ratings_df.to_csv(save_path, index=False)

    print("Predictions completed and saved.")




