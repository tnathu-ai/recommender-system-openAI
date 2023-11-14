import numpy as np
import openai
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import time
from constants import *
from evaluation_utils import *
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken

# OpenAI API Key
openai.api_key = OPENAI_API_KEY
AMAZON_CONTENT_SYSTEM = "Amazon Beauty products critic"

# Retry decorator configuration up to STOP_AFTER_N_ATTEMPTS times with an exponential backoff delay (1 to 20 seconds) between attempts.
# source: https://tenacity.readthedocs.io/en/latest/
retry_decorator = retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(STOP_AFTER_N_ATTEMPTS))

TOKENIZER = tiktoken.get_encoding(EMBEDDING_ENCODING)

# source: openai-cookbook
def check_and_reduce_length(text, 
                            max_tokens=MAX_TOKENS_CHAT_GPT, 
                            tokenizer=TOKENIZER):
    """
    Check and reduce the length of the text to be within the max_tokens limit.
    
    Parameters:
    text (str): The input text.
    max_tokens (int): Maximum allowed tokens.
    tokenizer: The tokenizer used for token counting.

    Returns:
    str: The text truncated to the max_tokens limit.
    """
    # Tokenize the text and check its length
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # If tokens exceed max_tokens, truncate the text
    truncated_text = ''
    for token in tokens[:max_tokens]:
        truncated_text += tokenizer.decode([token])

    return truncated_text

# function to extract the numeric rating from the response text
def extract_numeric_rating(rating_text):
    """Extract numeric rating from response text."""
    try:
        rating = float(re.search(r'\d+', rating_text).group())
        if 1 <= rating <= 5:
            return rating
        raise ValueError("Rating out of bounds")
    except (ValueError, AttributeError):
        print(f"Unexpected response for the provided details: {rating_text}")
        return 0  # Default value for unexpected responses

def generate_combined_text_for_prediction(columns, *args):
    """Generates a combined text string from columns and arguments for prediction."""
    return ". ".join([f"{col}: {val}" for col, val in zip(columns, args)])

@retry_decorator
def predict_rating_combined_ChatCompletion(combined_text, 
                                           model=GPT_MODEL_NAME, 
                                           temperature=TEMPERATURE, 
                                           use_few_shot=False, 
                                           rating_history=None,
                                           seed=RANDOM_STATE
                                           ):
    """
    Predicts product ratings using either a zero-shot or few-shot approach with the GPT model.

    Parameters:
    - combined_text (str): A combined text string containing product attributes and values.
    - model (str): The GPT model version to use.
    - temperature (float): Sampling temperature for the model response.
    - use_few_shot (bool): Whether to use the few-shot approach.
    - rating_history (str, optional): The user's past rating history, required if use_few_shot is True.

    Returns:
    - float: Predicted rating for the product or 0 if the response is not valid or not answerable.
    """
    if use_few_shot and rating_history is None:
        raise ValueError("Rating history is required for the few-shot approach.")

    combined_text = check_and_reduce_length(combined_text, MAX_TOKENS_CHAT_GPT, TOKENIZER)

    prompt_base = "Try your best to concisely answer the question based on the details below, and if the question can't be answered based on the context, provide insights on what might influence the product's rating.\n\nContext: Product Details - {combined_text}\n\n---\n\nQuestion: Based on these details, what might influence the product's rating? (1 being lowest and 5 being highest)\nAnswer:"

    prompt = prompt_base.format(combined_text=combined_text) if not use_few_shot else prompt_base.format(combined_text=f"User's Rating History - {rating_history}. {combined_text}")

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

    rating_text = response.choices[0].message['content'].strip()

    if "I don't know" in rating_text.lower():
        print("Insights provided by the model: ", rating_text)
        return 0  # Or handle this case differently

    return extract_numeric_rating(rating_text)



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
    
    # List to store predicted ratings
    predicted_ratings = []

    # Loop through each row in the data
    for idx, row in data.iterrows():
        prediction_data = row[columns_for_prediction].values
        combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data)
        predicted_rating = predict_rating_combined_ChatCompletion(combined_text, use_few_shot=False)
        print(f"Predicted rating for {prediction_data}: {predicted_rating}")
        predicted_ratings.append(predicted_rating)

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
    predicted_ratings = []
    actual_ratings = []
    users = data['reviewerID'].unique()

    for idx, reviewerID in enumerate(users):
        user_data = data[data['reviewerID'] == reviewerID]
        if len(user_data) >= 5:
            train_data = user_data.sample(4, random_state=RANDOM_STATE)
            test_data = user_data.sample(obs_per_user, random_state=RANDOM_STATE) if obs_per_user else user_data.drop(train_data.index)

            for _, test_row in test_data.iterrows():
                rating_history_str = ', '.join([f"{row[columns_for_training[0]]} ({row['rating']} stars): {row[columns_for_training[1]]}" for _, row in train_data.iterrows()])
                prediction_data = test_row[columns_for_prediction].values
                combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data)
                predicted_rating = predict_rating_combined_ChatCompletion(combined_text, rating_history_str=rating_history_str, use_few_shot=True)
                predicted_ratings.append(predicted_rating)
                actual_ratings.append(test_row['rating'])

        if (idx + 1) % pause_every_n_users == 0:
            print(f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    predicted_ratings_df = pd.DataFrame({'few_shot_predicted_rating': predicted_ratings, 'actual_rating': actual_ratings})
    predicted_ratings_df.to_csv(save_path, index=False)
