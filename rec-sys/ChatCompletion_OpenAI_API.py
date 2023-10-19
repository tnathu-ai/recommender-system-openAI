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


def predict_ratings_zero_shot_and_save(data, columns_for_unique_pairs=['title'], pause_every_n_users=PAUSE_EVERY_N_USERS, sleep_time=SLEEP_TIME, save_path='../../data/amazon-beauty/reviewText_large_predictions_zero_shot.csv'):

    # Nested function to predict rating using both title and reviewText
    def predict_rating_zero_shot_with_review(*args):
        combined_text = ". ".join(args)
        return predict_rating_zero_shot_ChatCompletion(combined_text)

    # Iterate through the dataset and predict ratings
    predicted_ratings = []
    unique_pairs = data[columns_for_unique_pairs].drop_duplicates().values
    for idx, unique_pair in enumerate(unique_pairs):
        predicted_rating = predict_rating_zero_shot_with_review(*unique_pair)
        print(f"Predicted rating for {unique_pair}: {predicted_rating}")
        predicted_ratings.append(predicted_rating)

        # Pause every pause_every_n_users rows
        if (idx + 1) % pause_every_n_users == 0:
            print(f"Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Create a DataFrame with columns_for_unique_pairs and predicted ratings
    columns_dict = {columns_for_unique_pairs[i]: unique_pairs[:, i] for i in range(
        len(columns_for_unique_pairs))}
    columns_dict['zero_shot_predicted_rating'] = predicted_ratings
    predicted_ratings_df = pd.DataFrame(columns_dict)

    # Merge the predicted ratings with the original data
    merged_data_with_predictions = pd.merge(
        data, predicted_ratings_df, on=columns_for_unique_pairs)

    # Save the merged data with predictions to the provided path
    merged_data_with_predictions.to_csv(save_path, index=False)


def predict_ratings_few_shot_and_save(data, columns_for_unique_pairs=['title'], obs_per_user=None, pause_every_n_users=PAUSE_EVERY_N_USERS, sleep_time=SLEEP_TIME, save_path='../../data/amazon-beauty/reviewText_small_predictions_few_shot.csv'):
    """
    Predict product ratings using the few-shot method and save the predictions to a CSV file.

    Parameters:
    - data: DataFrame containing the product reviews data.
    - columns_for_unique_pairs: List of columns to be used for unique product identification. 
      Default is ['title'].
    - obs_per_user: Number of observations to use for the test set per user. If None, all available data for the user will be used.
      Default is None.
    - pause_every_n_users: Number of users to process before pausing. Default value set by the PAUSE_EVERY_N_USERS constant.
    - sleep_time: Time to sleep in seconds after processing a batch of users. Default value set by the SLEEP_TIME constant.
    - save_path: Path to save the predicted ratings CSV file.

    Returns:
    None
    """

    # Nested function to predict rating using both title and reviewText with user's rating history
    def predict_rating_few_shot_with_review(*args, rating_history_str):
        combined_text = ". ".join(args)
        return predict_rating_few_shot_ChatCompletion(combined_text, rating_history_str)

    predicted_ratings = []
    actual_ratings = []

    # For each user in the dataset
    users = data['reviewerID'].unique()
    for idx, reviewerID in enumerate(users):
        user_data = data[data['reviewerID'] == reviewerID]

        # Check if the user has at least 5 ratings
        if len(user_data) >= 5:
            train_data = user_data.sample(4, random_state=RANDOM_STATE)
            if obs_per_user:
                test_data = user_data.sample(
                    obs_per_user, random_state=RANDOM_STATE)
            else:
                test_data = user_data.drop(train_data.index)

            # For each product in the testing set, use the training data to predict a rating
            for _, test_row in test_data.iterrows():
                rating_history_str = ', '.join(
                    [f"{row[columns_for_unique_pairs[0]]} ({row['rating']} stars): {row[columns_for_unique_pairs[1]]}" for _, row in train_data.iterrows()])
                predicted_rating = predict_rating_few_shot_with_review(
                    *test_row[columns_for_unique_pairs].values, rating_history_str=rating_history_str)

                predicted_ratings.append(predicted_rating)
                actual_ratings.append(test_row['rating'])

        # Introduce a pause after processing every pause_every_n_users
        if (idx + 1) % pause_every_n_users == 0:
            print(
                f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Save the predicted ratings to the provided path
    predicted_ratings_df = pd.DataFrame({
        'few_shot_predicted_rating': predicted_ratings,
        'actual_rating': actual_ratings
    })
    predicted_ratings_df.to_csv(save_path, index=False)
