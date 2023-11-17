import numpy as np
import openai
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import time
from constants import *
from evaluation_utils import *
from utils import *
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


@retry_decorator
def predict_rating_combined_ChatCompletion(combined_text, model=GPT_MODEL_NAME, temperature=TEMPERATURE, use_few_shot=False, rating_history=None, seed=RANDOM_STATE):
    """
    Predicts product ratings using either a zero-shot or few-shot approach with the GPT model.
    """
    if use_few_shot and rating_history is None:
        raise ValueError("Rating history is required for the few-shot approach.")

    # Check and reduce length of combined_text and rating_history
    combined_text = check_and_reduce_length(combined_text, MAX_TOKENS_CHAT_GPT // 2, TOKENIZER)
    if use_few_shot:
        rating_history = check_and_reduce_length(rating_history, MAX_TOKENS_CHAT_GPT // 2, TOKENIZER)

    # Construct the prompt
    prompt = f"Context: Product Details - {combined_text}"
    if use_few_shot:
        prompt += f"\nUser's Rating History - {rating_history}"
    prompt += "\n\n---\n\nQuestion: Based on these details, what might influence the product's rating? (1 being lowest and 5 being highest)\nAnswer:"

    # Create the API call
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=MAX_TOKENS_CHAT_GPT,
        seed=seed,
        messages=[
            {"role": "system", "content": AMAZON_CONTENT_SYSTEM},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the system fingerprint and print it
    system_fingerprint = response.get('system_fingerprint')
    print(f"System Fingerprint: {system_fingerprint}")

    # Extract and return the rating
    rating_text = response.choices[0].message['content'].strip()
    if "I don't know" in rating_text.lower():
        print("Insights provided by the model: ", rating_text)
        return 0
    return extract_numeric_rating(rating_text)


def predict_ratings_zero_shot_and_save(data,
                                       columns_for_prediction=['title'],
                                       title_column_name='title',  # Parameter for title column name
                                       asin_column_name=None,      # Parameter for ASIN column name, optional
                                       pause_every_n_users=PAUSE_EVERY_N_USERS,
                                       sleep_time=SLEEP_TIME,
                                       save_path='../../data/amazon-beauty/reviewText_large_predictions_zero_shot.csv'):
    predicted_ratings = []

    for idx, row in data.iterrows():
        prediction_data = row[columns_for_prediction].values
        combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data)
        predicted_rating = predict_rating_combined_ChatCompletion(combined_text, use_few_shot=False)

        product_title = row[title_column_name]
        product_details = f"{product_title}"

        if asin_column_name and asin_column_name in row:
            product_code = row[asin_column_name]
            product_details += f" (Code: {product_code})"

        print(f"Processing item {idx + 1}/{len(data)}\n")
        print(f"Details: {product_details}")
        print(f"\nPredicted Rating: {predicted_rating} stars")
        print(f"\n----------------\n")

        predicted_ratings.append(predicted_rating)

        if (idx + 1) % pause_every_n_users == 0:
            print(f"Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    data['zero_shot_predicted_rating'] = predicted_ratings
    data.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")



# Define the function to predict ratings using a few-shot approach
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
        
        if len(user_data) >= 5:
            # Sample training data
            train_data = user_data.sample(4, random_state=RANDOM_STATE)
            
            # Exclude the items in train_data from the test set
            test_data = user_data[~user_data.index.isin(train_data.index)]
            
            # If specified, limit the number of observations per user for the test set
            if obs_per_user:
                test_data = test_data.sample(obs_per_user, random_state=RANDOM_STATE)

            for test_idx, test_row in test_data.iterrows():
                # Generate combined text for prediction without the rating column
                prediction_data = {col: test_row[col] for col in columns_for_prediction if col != 'rating'}
                combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data.values())

                rating_history_str = ', '.join([f"{row[columns_for_training[0]]} ({row['rating']} stars)" for _, row in train_data.iterrows()])
                predicted_rating = predict_rating_combined_ChatCompletion(combined_text, rating_history=rating_history_str, use_few_shot=True)

                product_title = test_row.get(title_column_name, "Unknown Title")
                product_details = f"{product_title}"
                if asin_column_name and asin_column_name in test_row:
                    product_code = test_row[asin_column_name]
                    product_details += f" (Code: {product_code})"

                print(f"Processing user {idx + 1}/{len(users)}, item {test_idx + 1}/{len(test_data)}\n")
                print(f"User {user_id}:")
                print(f"Rating History for Prediction: {rating_history_str}")
                print(f"Predicted Item: {product_details}")
                print(f"Predicted Rating: {predicted_rating} stars")
                print(f"\n----------------\n")

                predicted_ratings.append(predicted_rating)
                actual_ratings.append(test_row['rating'])

        if (idx + 1) % pause_every_n_users == 0:
            print(f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    predicted_ratings_df = pd.DataFrame({'few_shot_predicted_rating': predicted_ratings, 'actual_rating': actual_ratings})
    predicted_ratings_df.to_csv(save_path, index=False)


# User based Collaborative Filtering
def predict_ratings_with_collaborative_filtering_and_save(data, interaction_matrix, user_mapper, item_mapper, user_inv_mapper, model_knn,
                                                          columns_for_training, columns_for_prediction,
                                                          user_column_name='reviewerID', title_column_name='title', asin_column_name='asin',
                                                          obs_per_user=None, pause_every_n_users=PAUSE_EVERY_N_USERS, sleep_time=SLEEP_TIME,
                                                          save_path='../../data/amazon-beauty/similar_users_predictions.csv'):
    predicted_ratings = []
    actual_ratings = []
    users = data[user_column_name].unique()

    for idx, user_id in enumerate(users):
        user_data = data[data[user_column_name] == user_id]

        if len(user_data) >= 5:
            test_data = user_data.sample(obs_per_user if obs_per_user else 1, random_state=RANDOM_STATE)
            remaining_data = user_data[~user_data.index.isin(test_data.index)]

            for test_idx, test_row in test_data.iterrows():
                user_idx = user_mapper[user_id]
                distances, indices = model_knn.kneighbors(interaction_matrix[user_idx], n_neighbors=5)

                training_data = pd.DataFrame()
                similar_users = []
                for idx in indices.flatten():
                    similar_user_id = user_inv_mapper[idx]
                    if similar_user_id != user_id:
                        similar_users.append(similar_user_id)
                        similar_user_data = data[data[user_column_name] == similar_user_id]
                        training_data = pd.concat([training_data, similar_user_data], ignore_index=True)

                # Log similar users
                print(f"Similar users for {user_id}: {similar_users}")

                prediction_data = {col: test_row[col] for col in columns_for_prediction if col != 'rating'}
                combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data.values())

                rating_history_str = ', '.join([f"{row[columns_for_training[0]]} ({row['rating']} stars)" for _, row in training_data.iterrows()])
                predicted_rating = predict_rating_combined_ChatCompletion(combined_text, rating_history=rating_history_str, use_few_shot=True)

                predicted_ratings.append(predicted_rating)
                actual_ratings.append(test_row['rating'])

                print(f"Processing user {idx + 1}/{len(users)}, item {test_idx + 1}/{len(test_data)}")
                print(f"User {user_id}:")
                print(f"Rating History for Prediction: {rating_history_str}")
                print(f"Predicted Item: {test_row[title_column_name]}")
                print(f"Predicted Rating: {predicted_rating} stars")
                print("\n----------------\n")

            if (idx + 1) % pause_every_n_users == 0:
                print(f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
                time.sleep(sleep_time)

    predicted_ratings_df = pd.DataFrame({'predicted_rating': predicted_ratings, 'actual_rating': actual_ratings})
    predicted_ratings_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")




