import numpy as np
import pandas as pd
import re
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from constants import *
from tenacity import retry, stop_after_attempt, wait_random_exponential

import openai
# Configure OpenAI API
openai.api_key = OPENAI_API_KEY


# Configured to retry up to STOP_AFTER_N_ATTEMPTS with an exponential backoff delay
retry_decorator = retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(STOP_AFTER_N_ATTEMPTS))

# Tokenizer setup for text processing
TOKENIZER = tiktoken.get_encoding(EMBEDDING_ENCODING)

def check_and_reduce_length(text, max_tokens=MAX_TOKENS_CHAT_GPT, tokenizer=TOKENIZER):
    """
    Check and reduce the length of the text to be within the max_tokens limit.

    Args:
        text (str): The input text.
        max_tokens (int): Maximum allowed tokens.
        tokenizer: The tokenizer used for token counting.

    Returns:
        str: The text truncated to the max_tokens limit.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_text = ''
    for token in tokens[:max_tokens]:
        truncated_text += tokenizer.decode([token])

    return truncated_text


def extract_numeric_rating(rating_text):
    """
    Extract numeric rating from response text.

    Args:
        rating_text (str): Text containing numeric rating.

    Returns:
        float: Extracted rating value. Returns 0 for unexpected responses.
    """
    try:
        # Updated regex to capture standalone digit rating
        rating_match = re.search(r'(\d+(\.\d+)?)', rating_text)
        if rating_match:
            rating = float(rating_match.group(1))
            if 1 <= rating <= 5:
                return rating
            else:
                print(f"Rating out of expected range (1-5): {rating_text}")
                return 0
        else:
            print(f"No valid rating found in the response: {rating_text}")
            return 0
    except Exception as e:
        print(f"Error extracting rating: {e}. Full response: {rating_text}")
        return 0


def generate_combined_text_for_prediction(columns, *args):
    """
    Generates a combined text string from columns and arguments for prediction.

    Args:
        columns (list): List of column names.
        args (tuple): Values corresponding to the columns.

    Returns:
        str: Combined text string for prediction.
    """
    return ". ".join([f"{col}: {val}" for col, val in zip(columns, args)])


def rerun_failed_predictions(data, failed_indices, prediction_function, save_path, **kwargs):
    # Log the number of failed indices
    print(f"Re-running predictions for {len(failed_indices)} failed cases.")
    
    # Filter the data to get only failed rows
    failed_data = data.iloc[failed_indices]
    
    # Check if failed_data is empty or not as expected
    print(f"Number of rows in failed data: {len(failed_data)}")

    # Re-run prediction on failed data
    failed_data = prediction_function(failed_data, **kwargs)

    # Log data shape after prediction
    print(f"Data shape after re-running predictions: {failed_data.shape}")
    
    # Update the original dataframe with new predictions
    data.loc[failed_indices] = failed_data

    # Save the updated dataframe
    data.to_csv(save_path, index=False)
    return data
