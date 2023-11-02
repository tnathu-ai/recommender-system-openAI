import numpy as np
import openai
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import time
from constants import *
from evaluation_utils import *
from tenacity import retry, wait_random_exponential, stop_after_attempt

# OpenAI API Key
openai.api_key = OPENAI_API_KEY


# Decorator to retry the function up to 6 times with an exponential backoff delay (1 to 20 seconds) between attempts.
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def predict_rating_zero_shot_ChatCompletion(title, model=GPT_MODEL_NAME, temperature=TEMPERATURE):
    prompt = f"How will users rate this product title: '{title}'? (1 being lowest and 5 being highest) Attention! Just give me back the exact whole number as a result, and you don't need a lot of text."

    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "You are an Amazon Beauty products critic."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    rating_text = response.choices[0].message['content'].strip()
    try:
        # Extract the first numerical value from the response
        # Only capture whole numbers
        rating = float(re.search(r'\d+', rating_text).group())
        if not (1 <= rating <= 5):
            raise ValueError("Rating out of bounds")
    except (ValueError, AttributeError):
        print(f"Unexpected response for '{title}': {rating_text}")
        rating = 0  # Set default value to 0 for unexpected responses

    return rating


# Decorator to retry the function up to 6 times with an exponential backoff delay (1 to 20 seconds) between attempts.
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def predict_rating_few_shot_ChatCompletion(product_title, rating_history, model=GPT_MODEL_NAME, temperature=TEMPERATURE):
    """
    Predict the rating of a product based on user's past rating history using the GPT model.

    Parameters:
    - product_title (str): The title of the product for which rating needs to be predicted.
    - rating_history (str): A string representation of user's past product ratings.
    - model (str): The GPT model version to use.
    - temperature (float): Sampling temperature for the model response.

    Returns:
    - float: Predicted rating for the product or None if the response is not valid.
    """
    # Construct the prompt to ask the model
    prompt = (f"Here is the user's rating history: {rating_history}. "
              f"Based on the above rating history, how many stars would you rate the product: '{product_title}'? "
              "(Provide a number between 1 and 5, either followed by the word 'stars' or preceded by the words 'would be'). "
              "Attention! Keep the response concise.")
    # Make the API call
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a product critic."},
            {"role": "user", "content": prompt}
        ]
    )

    rating_text = response.choices[0].message['content'].strip()
    try:
        # Extract the numerical value that appears close to "rating" or "stars"
        match = re.search(
            r'(\d+(\.\d*)?)\s*(?=stars|star)|rating\s*(\d+(\.\d*)?)|would be\s*(\d+(\.\d*)?)', rating_text)
        if match:
            # Extract the first matched group that's not None
            rating = float(next((m for m in match.groups() if m), 0))
        else:
            rating = 0
        if not (0.5 <= rating <= 5.0):
            raise ValueError("Rating out of bounds")
    except (ValueError, AttributeError):
        print(f"Unexpected response for '{product_title}': {rating_text}")
        rating = 0  # Set default value to 0 for unexpected responses

    return rating


def predict_ratings_zero_shot_and_save(data,
                                       columns_for_training=['title'],
                                       columns_for_prediction=['title'],
                                       pause_every_n_users=PAUSE_EVERY_N_USERS,
                                       sleep_time=SLEEP_TIME,
                                       save_path='../../data/amazon-beauty/reviewText_large_predictions_zero_shot.csv'):
    """
    This function predicts product ratings using a zero-shot approach and saves the predictions to a specified CSV file.

    Parameters:
    - data: DataFrame containing the product reviews data.
    - columns_for_training: List of columns used to uniquely identify each product for training. Default is ['title'].
    - columns_for_prediction: List of columns used for predicting ratings. Default is ['title'].
    - pause_every_n_users: The function will pause for a specified duration after processing a given number of users. This can be useful to avoid hitting API rate limits.
    - sleep_time: Duration (in seconds) for which the function should pause.
    - save_path: Path where the predictions will be saved as a CSV file.
    """

    # Nested function to predict ratings by combining provided arguments into a single string
    def predict_rating_zero_shot_with_review(*args):
        """Combine the arguments into a text and use them to predict the rating."""
        combined_text = ". ".join(map(str, args))
        return predict_rating_zero_shot_ChatCompletion(combined_text)

    # List to store predicted ratings
    predicted_ratings = []

    # Get unique pairs of products based on columns_for_training
    unique_pairs_for_training = data[columns_for_training].drop_duplicates(
    ).values

    # Loop through each unique product pair
    for idx, unique_pair in enumerate(unique_pairs_for_training):

        # Extract the row in the original data that matches the current unique pair
        mask = (data[columns_for_training].values == unique_pair).all(axis=1)
        matching_rows = data[mask]

        # Proceed only if there are matching rows
        if not matching_rows.empty:
            matching_row = matching_rows.iloc[0]

            # Get the data used for prediction
            prediction_data = matching_row[columns_for_prediction].values

            # Predict the rating
            predicted_rating = predict_rating_zero_shot_with_review(
                *prediction_data)
            print(f"Predicted rating for {unique_pair}: {predicted_rating}")

            # Append the predicted rating to the list
            predicted_ratings.append(predicted_rating)
        else:
            print(f"No matching data found for {unique_pair}. Skipping...")
            continue

        # Pause for sleep_time seconds after processing pause_every_n_users products
        if (idx + 1) % pause_every_n_users == 0:
            print(f"Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Create a DataFrame to store the unique product pairs and their predicted ratings
    columns_dict = {columns_for_training[i]: unique_pairs_for_training[:, i] for i in range(
        len(columns_for_training))}
    columns_dict['zero_shot_predicted_rating'] = predicted_ratings
    predicted_ratings_df = pd.DataFrame(columns_dict)

    # Merge the original data with the predicted ratings
    merged_data_with_predictions = pd.merge(
        data, predicted_ratings_df, on=columns_for_training)

    # Save the merged data with predictions to the specified path
    merged_data_with_predictions.to_csv(save_path, index=False)


def predict_ratings_few_shot_and_save(data,
                                      columns_for_training=['title'],
                                      columns_for_prediction=['title'],
                                      obs_per_user=None,
                                      pause_every_n_users=PAUSE_EVERY_N_USERS,
                                      sleep_time=SLEEP_TIME,
                                      save_path='../../data/amazon-beauty/reviewText_small_predictions_few_shot.csv'):
    """
    This function predicts product ratings using a few-shot approach, which utilizes a user's rating history, 
    and then saves the predictions to a specified CSV file.

    Parameters:
    - data: DataFrame containing the product reviews data.
    - columns_for_training: List of columns used to uniquely identify each product for training. Default is ['title'].
    - columns_for_prediction: List of columns used for predicting ratings. Default is ['title'].
    - obs_per_user: Number of observations to use for the test set per user. If None, all available data for the user will be used.
    - pause_every_n_users: The function will pause for a specified duration after processing a given number of users. This can be useful to avoid hitting API rate limits.
    - sleep_time: Duration (in seconds) for which the function should pause.
    - save_path: Path where the predictions will be saved as a CSV file.
    """

    # Nested function to predict ratings by combining provided arguments into a single string along with rating history
    def predict_rating_few_shot_with_review(*args, rating_history_str):
        """Combine the arguments into a text and use them along with rating history to predict the rating."""
        combined_text = ". ".join(map(str, args))
        return predict_rating_few_shot_ChatCompletion(combined_text, rating_history_str)

    # Lists to store predicted ratings and actual ratings
    predicted_ratings = []
    actual_ratings = []

    # Get unique users from the data
    users = data['reviewerID'].unique()

    # Loop through each user
    for idx, reviewerID in enumerate(users):
        # Extract the data related to the current user
        user_data = data[data['reviewerID'] == reviewerID]

        # Ensure the user has at least 5 ratings
        if len(user_data) >= 5:
            # Randomly sample 4 rows for training
            train_data = user_data.sample(4, random_state=RANDOM_STATE)

            # Depending on the value of obs_per_user, sample rows for testing
            if obs_per_user:
                test_data = user_data.sample(
                    obs_per_user, random_state=RANDOM_STATE)
            else:
                test_data = user_data.drop(train_data.index)

            # Loop through each row in the test data
            for _, test_row in test_data.iterrows():
                # Create a rating history string using the training data
                rating_history_str = ', '.join(
                    [f"{row[columns_for_training[0]]} ({row['rating']} stars): {row[columns_for_training[1]]}" for _, row in train_data.iterrows()])

                # Extract the data used for prediction from the test row
                prediction_data = test_row[columns_for_prediction].values

                # Predict the rating using the few-shot approach
                predicted_rating = predict_rating_few_shot_with_review(
                    *prediction_data, rating_history_str=rating_history_str)

                # Append the predicted rating and the actual rating to their respective lists
                predicted_ratings.append(predicted_rating)
                actual_ratings.append(test_row['rating'])

        # Introduce a pause after processing a set number of users
        if (idx + 1) % pause_every_n_users == 0:
            print(
                f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Save the predicted and actual ratings to the specified path
    predicted_ratings_df = pd.DataFrame({
        'few_shot_predicted_rating': predicted_ratings,
        'actual_rating': actual_ratings
    })
    predicted_ratings_df.to_csv(save_path, index=False)
