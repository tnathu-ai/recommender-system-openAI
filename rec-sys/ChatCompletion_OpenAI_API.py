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


# Decorator to retry the function up to STOP_AFTER_N_ATTEMPTS times with an exponential backoff delay (1 to 20 seconds) between attempts.
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(STOP_AFTER_N_ATTEMPTS))
def predict_rating_zero_shot_ChatCompletion(combined_text, model=GPT_MODEL_NAME, temperature=TEMPERATURE):
    """
    Predicts product ratings using a zero-shot approach with the GPT model and a combined text prompt.

    Parameters:
    - combined_text (str): A combined text string containing product attributes and values.
    - model (str): The GPT model version to use.
    - temperature (float): Sampling temperature for the model response.

    Returns:
    - float: Predicted rating for the product or 0 if the response is not valid or not answerable.
    """

    # Construct the prompt for the model
    prompt = (f"Answer the question based on the details below, and if the question can't be answered based on the context, "
              f"say \"I don't know\"\n\nContext: Product Details - {combined_text}\n\n---\n\n"
              f"Question: How will users rate this product based on these details? (1 being lowest and 5 being highest)\nAnswer:")

    # Make the API call
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=150,  # Set a limit to the number of tokens to generate
        messages=[
            {"role": "system", "content": "You are an Amazon Beauty products critic."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content from the response
    rating_text = response.choices[0].message['content'].strip()

    # Check if the model's response is "I don't know"
    if rating_text.lower() == "i don't know":
        print("Unable to determine a rating based on the provided context.")
        return 0

    try:
        # Extract the first numerical value from the response and ensure it's a whole number
        rating = float(re.search(r'\d+', rating_text).group())
        
        # Validate if the rating is within the expected bounds (1 to 5)
        if not (1 <= rating <= 5):
            raise ValueError("Rating out of bounds")
    except (ValueError, AttributeError):
        # Handle unexpected responses or errors
        print(f"Unexpected response for the provided details: {rating_text}")
        rating = 0  # Set default value to 0 for unexpected responses

    # Return the predicted rating
    return rating


# Decorator to retry the function up to STOP_AFTER_N_ATTEMPTS times with an exponential backoff delay (1 to 20 seconds) between attempts.
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(STOP_AFTER_N_ATTEMPTS))
def predict_rating_few_shot_ChatCompletion(combined_text, rating_history, model=GPT_MODEL_NAME, temperature=TEMPERATURE):
    """
    Predict the rating of a product based on user's past rating history and combined textual information using the GPT model.

    Parameters:
    - combined_text (str): A string containing combined information about the product, which may include title, description, or other relevant details.
    - rating_history (str): A string representation of user's past product ratings.
    - model (str): The GPT model version to use.
    - temperature (float): Sampling temperature for the model response.

    Returns:
    - float: Predicted rating for the product or 0 if the response is not valid or not answerable.
    """
    # Construct the prompt to ask the model
    prompt = (f"Answer the question based on the user's rating history and the product details below, "
              f"and if the question can't be answered based on the context, say \"I don't know\"\n\n"
              f"Context: User's Rating History - {rating_history}. Product Details - {combined_text}\n\n---\n\n"
              f"Question: How many stars would you rate the product?\nAnswer:")

    # Make the API call
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=150,  # Set a limit to the number of tokens to generate
        messages=[
            {"role": "system", "content": "You are a product critic."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content from the response
    rating_text = response.choices[0].message['content'].strip()

    # Check if the model's response is "I don't know"
    if rating_text.lower() == "i don't know":
        print("Unable to determine a rating based on the provided context.")
        return 0

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
        print(f"Unexpected response for '{combined_text}': {rating_text}")
        rating = 0  # Set default value to 0 for unexpected responses

    return rating


def predict_ratings_zero_shot_and_save(data,
                                       columns_for_prediction=['title'],
                                       pause_every_n_users=PAUSE_EVERY_N_USERS,
                                       sleep_time=SLEEP_TIME,
                                       save_path='../../data/amazon-beauty/reviewText_large_predictions_zero_shot.csv'):
    """
    This function predicts product ratings using a zero-shot approach and saves the predictions to a specified CSV file.

    Parameters:
    - data: DataFrame containing the product reviews data.
    - columns_for_prediction: List of columns used for predicting ratings. Default is ['title'].
    - pause_every_n_users: The function will pause for a specified duration after processing a given number of users. This can be useful to avoid hitting API rate limits.
    - sleep_time: Duration (in seconds) for which the function should pause.
    - save_path: Path where the predictions will be saved as a CSV file.
    """

    # Nested function to predict ratings by combining provided arguments into a single string
    def predict_rating_zero_shot_with_review(*args):
        """Combine the arguments into a text and use them to predict the rating."""
        combined_text = ". ".join([f"{columns_for_prediction[i]}: {args[i]}" for i in range(len(args))])
        return predict_rating_zero_shot_ChatCompletion(combined_text)

    # List to store predicted ratings
    predicted_ratings = []

    # Loop through each row in the data
    for idx, row in data.iterrows():

        # Get the data used for prediction
        prediction_data = row[columns_for_prediction].values

        # Predict the rating
        predicted_rating = predict_rating_zero_shot_with_review(*prediction_data)
        print(f"Predicted rating for {prediction_data}: {predicted_rating}")

        # Append the predicted rating to the list
        predicted_ratings.append(predicted_rating)

        # Pause for sleep_time seconds after processing pause_every_n_users products
        if (idx + 1) % pause_every_n_users == 0:
            print(f"Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Add the predicted ratings to the original data
    data['zero_shot_predicted_rating'] = predicted_ratings

    # Save the data with predictions to the specified path
    data.to_csv(save_path, index=False)



# Define the function to predict ratings using a few-shot approach
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
        combined_text = ". ".join([f"{columns_for_prediction[i]}: {arg}" for i, arg in enumerate(args)])
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
                test_data = user_data.sample(obs_per_user, random_state=RANDOM_STATE)
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

                # Print the predicted rating
                if predicted_rating:
                    print(f"Predicted rating for {test_row[columns_for_training[0]]}: {predicted_rating}")
                else:
                    print(f"No rating predicted for {test_row[columns_for_training[0]]}")

                # Append the predicted rating and the actual rating to their respective lists
                predicted_ratings.append(predicted_rating)
                actual_ratings.append(test_row['rating'])

        # Introduce a pause after processing a set number of users
        if (idx + 1) % pause_every_n_users == 0:
            print(f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Save the predicted and actual ratings to the specified path
    predicted_ratings_df = pd.DataFrame({
        'few_shot_predicted_rating': predicted_ratings,
        'actual_rating': actual_ratings
    })
    predicted_ratings_df.to_csv(save_path, index=False)
