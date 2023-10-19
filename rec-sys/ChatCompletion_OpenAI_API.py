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


# Function to predict rating using both title and reviewText
def predict_rating_zero_shot_with_review(title, review):
    return predict_rating_zero_shot_ChatCompletion(f"{title}. {review}")


# Function to predict rating using both title and reviewText with user's rating history
def predict_rating_few_shot_with_review(title, review, rating_history_str):
    return predict_rating_few_shot_ChatCompletion(f"{title}. {review}", rating_history_str)
