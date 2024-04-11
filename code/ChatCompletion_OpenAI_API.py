import numpy as np
import openai
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import time
from constants import *
from evaluation_utils import *
from user_utils import *
from utils import *
from CF_utils import *
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import openai
from openai.error import APIError, APIConnectionError, RateLimitError

# Configure agent
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
    
    # Initialize prompt variable
    prompt = ""

    # Check and reduce length of combined_text
    combined_text = check_and_reduce_length(combined_text, MAX_TOKENS_CHAT_GPT // 3, TOKENIZER)

    # Construct the prompt based on the approach
    if approach == "few-shot":
        rating_history = check_and_reduce_length(rating_history, MAX_TOKENS_CHAT_GPT // 3, TOKENIZER)
        prompt += f"\n\nHere is user rating history:\n{rating_history}"
        prompt += f"\n\nBased on above rating history, please predict user's rating for the product {combined_text}, (1 being lowest and 5 being highest,The output should be like: (x stars, xx%), do not explain the reason.)"

    elif approach == "CF":
        rating_history = check_and_reduce_length(rating_history, MAX_TOKENS_CHAT_GPT // 3, TOKENIZER)
        prompt += f"\n\nHere is user rating history:\n{rating_history}"
        similar_users_ratings = check_and_reduce_length(similar_users_ratings, MAX_TOKENS_CHAT_GPT // 3, TOKENIZER)
        prompt += f"\n\nHere is the rating history from users who are similar to this user:\n{similar_users_ratings}"
        prompt += f"\n\nBased on above rating history and similar users' rating history, please predict user's rating for the product {combined_text}, (1 being lowest and 5 being highest,The output should be like: (x stars, xx%), do not explain the reason.)"
        
    else:
        prompt = f"How will user rate this product {combined_text}? (1 being lowest and 5 being highest) Attention! Just give me back the exact number as a result, and you don't need a lot of text."
        

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
        print(f"\nSystem Fingerprint: {system_fingerprint}")
        # Extract and return the rating
        rating_text = response.choices[0].message['content'].strip()
        print(f'\nAPI call response: "{rating_text}"')
        extracted_rating = extract_numeric_rating(rating_text)
        print(f'Extracted rating: {extracted_rating}\n\n\n')
        print("----------------------------------------------------------------------------------")
        return extracted_rating  # A float
    
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
                                       seed=RANDOM_STATE,
                                       system_content=AMAZON_CONTENT_SYSTEM):
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
        # Select test set for the user
        test_set, _ = select_test_set_for_user(user_data, num_tests=TEST_OBSERVATION_PER_USER, seed=seed)
        random_row = test_set.iloc[0]

        # Generate combined text for prediction using specified columns
        combined_text = ' | '.join([f"{col}: {random_row[col]}" for col in columns_for_prediction])

        # Predict rating using zero-shot approach
        predicted_rating = predict_rating_combined_ChatCompletion(combined_text, approach="zero-shot", system_content=system_content)
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
                                      save_path='few_shot_predictions.csv',
                                      system_content=AMAZON_CONTENT_SYSTEM):

    users = data[user_column_name].unique()
    print(f"Total number of users: {len(users)}")

    for idx, user_id in enumerate(users):
        user_data = data[data[user_column_name] == user_id]

        if len(user_data) < 5:
            continue

        # Select test set for the user
        test_set, remaining_data = select_test_set_for_user(user_data, num_tests=TEST_OBSERVATION_PER_USER, seed=RANDOM_STATE)
        if test_set.empty:
            continue
        test_row = test_set.iloc[0]

        # Use remaining_data for training
        train_data = remaining_data.head(4)  # Select 4 distinct previous ratings

        if len(train_data) < 4:
            continue  # Skip if there are not enough historical ratings

        prediction_data = {col: test_row[col] for col in columns_for_prediction if col != rating_column_name}
        combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data.values())
        print(f'Predicting rating for: "{combined_text}"')

        rating_history_str = '\n'.join([
            '* ' + ' | '.join(f"{col}: {row[col]}" for col in columns_for_training) + f" - Rating: {row[rating_column_name]} stars"
            for _, row in train_data.iterrows()
        ])
        print(f"Rating history:\n{rating_history_str}")

        predicted_rating = predict_rating_combined_ChatCompletion(combined_text, 
                                                                  rating_history=rating_history_str, 
                                                                  approach="few-shot", 
                                                                  system_content=system_content)

        item_id = test_row[asin_column_name]
        actual_rating = test_row[rating_column_name]
        title = test_row[title_column_name]

        results.append([user_id, item_id, title, actual_rating, predicted_rating])

   
        if (idx + 1) % pause_every_n_users == 0:
            print(f"Processed {idx + 1} users. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating'])
    results_df.to_csv(save_path, index=False)
    print("Predictions saved to", save_path)
    return results_df 



def rerun_failed_few_shot_predictions(data, 
                                      columns_for_training, 
                                      columns_for_prediction, 
                                      user_column_name, 
                                      title_column_name, 
                                      asin_column_name, 
                                      rating_column_name,
                                      pause_every_n_users, 
                                      sleep_time, 
                                      save_path, 
                                      new_path,
                                      rerun_indices,
                                      system_content=AMAZON_CONTENT_SYSTEM):
    # Load the original predictions
    original_data = pd.read_csv(save_path)
    

    # Define fixed column names for output
    fixed_columns = ['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating', 'is_rating_float']

    # Map dynamic column names to fixed column names
    column_mapping = {
        user_column_name: 'user_id',
        asin_column_name: 'item_id',
        title_column_name: 'title',
        rating_column_name: 'actual_rating'
    }

    # Rerun predictions for each failed index
    for idx in rerun_indices:
        user_id = original_data.loc[idx, column_mapping[user_column_name]]
        item_id = original_data.loc[idx, column_mapping[asin_column_name]]

        # Retrieve user's data
        user_data = data[data[user_column_name] == user_id]

        if len(user_data) < 5:
            print(f"Skipping User ID: {user_id} - Insufficient data")
            continue

        user_data = user_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        # Find the row that needs prediction
        test_row = user_data[user_data[asin_column_name] == item_id].iloc[0]

        # Skip the current item being predicted
        train_data = user_data[user_data[asin_column_name] != test_row[asin_column_name]]

        if len(train_data) < 4:
            print(f"Skipping User ID: {user_id}, Item ID: {item_id} - Not enough historical ratings")
            continue

        train_data = train_data.head(4)

        prediction_data = {col: test_row[col] for col in columns_for_prediction if col != rating_column_name}
        combined_text = generate_combined_text_for_prediction(columns_for_prediction, *prediction_data.values())

        rating_history_str = '\n'.join([
            '* ' + ' | '.join(f"{col}: {row[col]}" for col in columns_for_training) + f" - Rating: {row[rating_column_name]} stars"
            for _, row in train_data.iterrows()
        ])

        print(f"Rerunning prediction with historical ratings:\n{rating_history_str}\nPredicting rating for: '{combined_text}'")

        predicted_rating = predict_rating_combined_ChatCompletion(combined_text, 
                                                                  rating_history=rating_history_str, 
                                                                  approach="few-shot",
                                                                  system_content=system_content)

        # Update the prediction in the original dataframe
        original_data.loc[idx, 'predicted_rating'] = predicted_rating
        original_data.loc[idx, 'is_rating_float'] = pd.notna(pd.to_numeric(predicted_rating, errors='coerce'))

        print(f"Updated prediction for User ID: {user_id}, Item ID: {item_id}: {predicted_rating}")

        if (idx + 1) % pause_every_n_users == 0:
            print(f"Processed {idx + 1} indices. Pausing for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Save the updated dataframe with fixed column names
    original_data = original_data.rename(columns=column_mapping)
    original_data = original_data[fixed_columns]
    original_data.to_csv(new_path, index=False)
    print(f"Updated predictions saved to {new_path}")




def predict_ratings_with_collaborative_filtering_and_save(data, pcc_matrix, 
                                                          user_column_name='reviewerID', 
                                                          movie_column_name='title', 
                                                          movie_id_column='asin',
                                                          rating_column_name='rating', 
                                                          num_ratings_per_user=NUM_RATINGS_PER_USER, 
                                                          num_similar_users=NUM_SIMILAR_USERS,
                                                          num_main_user_ratings=NUM_MAIN_USER_RATINGS,
                                                          save_path='cf_predictions.csv', 
                                                          seed=RANDOM_STATE,
                                                          system_content=AMAZON_CONTENT_SYSTEM):
    results = []
    unique_users = data[user_column_name].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
    

    random.seed(seed)

    for user_id in unique_users:
        user_idx = user_id_to_index[user_id]

        print(f"Processing user {user_id} (Index: {user_idx})")

        # Retrieve the main user's historical ratings randomly
        main_user_data = data[data[user_column_name] == user_id]
        main_user_ratings = main_user_data.sample(n=num_main_user_ratings, random_state=seed)

        main_user_ratings_str = '\n'.join([
            f"* Title: {row[movie_column_name]}, Rating: {row[rating_column_name]} stars"
            for _, row in main_user_ratings.iterrows()
        ])

        # Find the top similar users based on Pearson Correlation Coefficient
        similar_users_idx = np.argsort(-pcc_matrix[user_idx])[:num_similar_users + 1]
        similar_users_idx = similar_users_idx[similar_users_idx != user_idx][:num_similar_users]

        print(f"Top similar users for {user_id}: {[unique_users[idx] for idx in similar_users_idx]}")

        # Collect historical ratings from similar users randomly
        similar_users_ratings = ""
        for idx in similar_users_idx:
            similar_user_id = unique_users[idx]
            similar_user_data = data[data[user_column_name] == similar_user_id]
            historical_ratings = similar_user_data.sample(n=num_ratings_per_user, random_state=seed)
            for _, row in historical_ratings.iterrows():
                rating_info = f"* Title: {row[movie_column_name]}, Rating: {row[rating_column_name]} stars"
                similar_users_ratings += rating_info + "\n"
                
        # List of movie IDs already rated by the user
        rated_movie_ids = main_user_ratings[movie_id_column].tolist()

        # Exclude already rated movies and select a random movie for prediction
        potential_movies_for_prediction = main_user_data[~main_user_data[movie_id_column].isin(rated_movie_ids)]
        if potential_movies_for_prediction.empty:
            print(f"No unrated movies available for user {user_id} for prediction.")
            continue

        random_movie_row = potential_movies_for_prediction.sample(n=1, random_state=seed).iloc[0]
        random_movie_title = random_movie_row[movie_column_name]
        random_movie_id = random_movie_row[movie_id_column]
        actual_rating = random_movie_row[rating_column_name]


        # Construct prompt for API call
        combined_text = f"Title: {random_movie_title}"
        prompt = f"Main User Ratings:\n{main_user_ratings_str}\n\nSimilar Users' Ratings:\n{similar_users_ratings}\n\nPredict rating for '{combined_text}':"

        print(f"Generated prompt for user {user_id}:\n{prompt}")

        predicted_rating = predict_rating_combined_ChatCompletion(
            combined_text, 
            approach="CF", 
            similar_users_ratings=similar_users_ratings,
            rating_history=main_user_ratings_str,
            system_content=system_content
        )

        # Store prediction results
        results.append([user_id, random_movie_id, random_movie_title, actual_rating, predicted_rating])

        print(f"User {user_id}: Predicted rating for '{random_movie_title}' is {predicted_rating}.")

    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating'])
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    return results_df



def rerun_failed_CF_fewshot_predictions(data, pcc_matrix, 
                                        save_path, 
                                        user_column_name, 
                                        movie_column_name, 
                                        movie_id_column, 
                                        rating_column_name, 
                                        num_ratings_per_user, 
                                        num_main_user_ratings, 
                                        num_similar_users, 
                                        new_path, 
                                        rerun_indices,
                                        system_content=AMAZON_CONTENT_SYSTEM):
    # Load original predictions and standardize column names
    original_data = pd.read_csv(save_path)
    original_data.columns = ['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating']
    
    # Map unique users to their index for quick access
    unique_users = data[user_column_name].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}

    # Process each failed prediction index
    for index in rerun_indices:
        user_id = original_data.at[index, 'user_id']
        item_id = original_data.at[index, 'item_id']
        user_idx = user_id_to_index[user_id]

        print(f"Rerunning prediction for User ID: {user_id}, Item ID: {item_id} (Index: {index})")

        # Retrieve the main user's historical ratings for context
        main_user_data = data[data[user_column_name] == user_id]
        # Exclude the failed movie item from the historical ratings
        main_user_data_without_failed_item = main_user_data[main_user_data[movie_id_column] != item_id]
        
        # If there are not enough ratings to sample from, skip this user
        if len(main_user_data_without_failed_item) < num_main_user_ratings:
            print(f"Not enough historical data to rerun prediction for User ID: {user_id}. Skipping.")
            continue

        main_user_ratings = main_user_data_without_failed_item.sample(n=num_main_user_ratings)
        main_user_ratings_str = '\n'.join([
            f"* Title: {row[movie_column_name]}, Rating: {row[rating_column_name]} stars"
            for _, row in main_user_ratings.iterrows()
        ])

        # Identify similar users based on Pearson Correlation Coefficients
        similar_users_idx = np.argsort(-pcc_matrix[user_idx])[:num_similar_users + 1]
        similar_users_idx = similar_users_idx[similar_users_idx != user_idx][:num_similar_users]

        # Compile historical ratings from similar users for context
        similar_users_ratings = ""
        for idx in similar_users_idx:
            similar_user_id = unique_users[idx]
            similar_user_data = data[data[user_column_name] == similar_user_id]
            historical_ratings = similar_user_data.sample(n=num_ratings_per_user)

            for _, row in historical_ratings.iterrows():
                rating_info = f"* Title: {row[movie_column_name]}, Rating: {row[rating_column_name]} stars"
                similar_users_ratings += rating_info + "\n"
                
        # List of movie IDs already rated by the user
        rated_movie_ids = main_user_ratings[movie_id_column].tolist()

        # Check if the failed movie is in the main user's historical ratings
        if item_id in rated_movie_ids:
            print(f"Failed movie {item_id} is in the main user's historical ratings. Skipping.")
            continue

        # Select the specific item that had a failed prediction
        failed_movie_row = data[(data[user_column_name] == user_id) & (data[movie_id_column] == item_id)].iloc[0]
        failed_movie_title = failed_movie_row[movie_column_name]

        # Construct text for re-prediction
        combined_text = f"Title: {failed_movie_title}"
        
        # Call prediction function with the constructed context
        predicted_rating = predict_rating_combined_ChatCompletion(
            combined_text, 
            approach="CF", 
            similar_users_ratings=similar_users_ratings,
            rating_history=main_user_ratings_str,
            system_content=system_content
        )

        # Update the predicted rating in the original data
        original_data.at[index, 'predicted_rating'] = predicted_rating

        print(f"Updated prediction for User ID: {user_id}, Item ID: {item_id}: {predicted_rating}")

    # Save the updated predictions to a new file
    original_data.to_csv(new_path, index=False)
    print(f"Updated predictions saved to {new_path}")
    
    


def predict_ratings_with_CF_item_and_save(data, user_pcc_matrix, item_pcc_matrix,
                                              user_column_name='reviewerID', 
                                              movie_column_name='title', 
                                              movie_id_column='asin',
                                              rating_column_name='rating', 
                                              timestamp_column_name='Timestamp', 
                                              num_ratings_per_user=NUM_RATINGS_PER_USER, 
                                              num_similar_users=NUM_SIMILAR_USERS,
                                              num_main_user_ratings=NUM_MAIN_USER_RATINGS,
                                              test_selection_method='random',
                                              save_path='cf_predictions.csv', 
                                              seed=RANDOM_STATE,
                                              system_content=AMAZON_CONTENT_SYSTEM):
    results = []

    unique_users = data[user_column_name].unique()
    unique_items = data[movie_id_column].unique()

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(unique_items)}

    random.seed(seed)

    for user_id in unique_users:
        user_idx = user_id_to_index[user_id]
        print(f"Processing user {user_id} (Index: {user_idx})")

        main_user_data = data[data[user_column_name] == user_id]
        
        # Select test set based on the specified method
        if test_selection_method == 'random':
            test_set, remaining_data = select_test_set_for_user(main_user_data, num_tests=TEST_OBSERVATION_PER_USER, seed=seed)
        elif test_selection_method == 'sequential':
            test_set, remaining_data = sequential_train_test_split(main_user_data, time_column=timestamp_column_name)


        if test_set.empty:
            print(f"No test data available for user {user_id}.")
            continue

        for random_movie_row in test_set.itertuples():
            random_movie_title = getattr(random_movie_row, movie_column_name)
            random_movie_id = getattr(random_movie_row, movie_id_column)
            random_movie_index = item_id_to_index.get(random_movie_id)
            actual_rating = getattr(random_movie_row, rating_column_name)

            if random_movie_index is None or random_movie_index >= item_pcc_matrix.shape[0]:
                print(f"Item '{random_movie_id}' not found or out of bounds in item_pcc_matrix.")
                continue

            if len(remaining_data) < num_main_user_ratings:
                main_user_ratings = remaining_data
            else:
                main_user_ratings = remaining_data.sample(n=num_main_user_ratings, random_state=seed)
            main_user_ratings_str = '\n'.join([
                f"* Title: {row[movie_column_name]}, Rating: {row[rating_column_name]} stars"
                for _, row in main_user_ratings.iterrows()
            ])

            similar_users_idx = np.argsort(-user_pcc_matrix[user_idx])[:num_similar_users + 1]
            similar_users_idx = similar_users_idx[similar_users_idx != user_idx][:num_similar_users]

            similar_users_ratings = ""
            for idx in similar_users_idx:
                similar_user_id = unique_users[idx]
                similar_user_data = data[data[user_column_name] == similar_user_id]

                # Find top-rated items by this similar user, sorted by item PCC
                similar_items_indices = np.argsort(-item_pcc_matrix[random_movie_index, :])
                top_rated_items = similar_user_data[similar_user_data[movie_id_column].isin(unique_items[similar_items_indices])]

                # Extract top ratings from this user
                top_ratings = top_rated_items.nlargest(num_ratings_per_user, rating_column_name)
                for top_rating_row in top_ratings.itertuples():
                    item_id = getattr(top_rating_row, movie_id_column)
                    rating = getattr(top_rating_row, rating_column_name)
                    item_title = data.loc[data[movie_id_column] == item_id, movie_column_name].iloc[0]  # Get the title of the item
                    similar_users_ratings += f"* Title: {item_title}, Rating: {rating} stars\n"

            combined_text = f"Title: {random_movie_title}"
            prompt = f"Main User Ratings:\n{main_user_ratings_str}\n\nSimilar Users' Ratings:\n{similar_users_ratings}\n\nPredict rating for '{combined_text}':"

            predicted_rating = predict_rating_combined_ChatCompletion(
                combined_text, 
                approach="CF", 
                similar_users_ratings=similar_users_ratings,
                rating_history=main_user_ratings_str,
                system_content=system_content
            )

            results.append([user_id, random_movie_id, random_movie_title, actual_rating, predicted_rating])
            print(f"User {user_id}: Predicted rating for '{random_movie_title}' is {predicted_rating}.")

    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating'])
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    return results_df


    



def rerun_failed_CF_item_PCC_predictions_sequential(data, user_pcc_matrix, item_pcc_matrix,
                                         save_path, user_column_name, movie_column_name,
                                         movie_id_column, rating_column_name,
                                         num_ratings_per_user, num_main_user_ratings, num_similar_users,
                                         new_path, rerun_indices, seed=RANDOM_STATE,
                                         system_content=AMAZON_CONTENT_SYSTEM):
    # Load the original predictions
    original_data = pd.read_csv(save_path)
    original_data.columns = ['user_id', 'item_id', 'title', 'actual_rating', 'predicted_rating']

    # Re-seed for reproducibility
    random.seed(seed)

    # Map unique users and items to their indices for quick access
    unique_users = data[user_column_name].unique()
    unique_items = data[movie_id_column].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(unique_items)}

    for index in rerun_indices:
        user_id = original_data.at[index, 'user_id']
        item_id = original_data.at[index, 'item_id']
        user_idx = user_id_to_index.get(user_id)
        item_idx = item_id_to_index.get(item_id)

        if user_idx is None or item_idx is None:
            print(f"User ID: {user_id} or Item ID: {item_id} not found in index. Skipping.")
            continue

        print(f"Rerunning prediction for User ID: {user_id}, Item ID: {item_id} (Index: {index})")

        # Retrieve user's and item's data
        user_data = data[data[user_column_name] == user_id]
        item_data = data[data[movie_id_column] == item_id]

        if item_data.empty:
            print(f"Item data for ID: {item_id} not found. Skipping.")
            continue

        # Sample user's historical ratings
        if len(user_data) < num_main_user_ratings:
            main_user_ratings = user_data
        else:
            main_user_ratings = user_data.sample(n=num_main_user_ratings, random_state=seed)

        # Construct the context from the user's ratings
        main_user_ratings_str = '\n'.join([
            f"* Title: {row[movie_column_name]}, Rating: {row[rating_column_name]} stars"
            for _, row in main_user_ratings.iterrows()
        ])

        # Identify similar users and items
        similar_users_idx = np.argsort(-user_pcc_matrix[user_idx])[:num_similar_users + 1]
        similar_users_idx = similar_users_idx[similar_users_idx != user_idx][:num_similar_users]

        similar_items_idx = np.argsort(-item_pcc_matrix[item_idx])[:num_similar_users + 1]
        similar_items_idx = similar_items_idx[similar_items_idx != item_idx][:num_similar_users]

        # Compile ratings from similar users and items
        similar_users_ratings = ""
        for idx in similar_users_idx:
            similar_user_id = unique_users[idx]
            similar_user_data = data[data[user_column_name] == similar_user_id]
            for _, row in similar_user_data.iterrows():
                similar_users_ratings += f"* Title: {row[movie_column_name]}, Rating: {row[rating_column_name]} stars\n"

        # Predict the rating
        combined_text = f"Title: {item_data.iloc[0][movie_column_name]}"
        prompt = f"Main User Ratings:\n{main_user_ratings_str}\n\nSimilar Users' Ratings:\n{similar_users_ratings}\n\nPredict rating for '{combined_text}':"
        predicted_rating = predict_rating_combined_ChatCompletion(
            combined_text, approach="CF", similar_users_ratings=similar_users_ratings,
            rating_history=main_user_ratings_str, system_content=system_content
        )

        # Update the original data with the new prediction
        original_data.at[index, 'predicted_rating'] = predicted_rating
        print(f"Updated prediction for User ID: {user_id}, Item ID: {item_id}: {predicted_rating}")

    # Save the updated predictions to a new file
    original_data.to_csv(new_path, index=False)
    print(f"Updated predictions saved to {new_path}")