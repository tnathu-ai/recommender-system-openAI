from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import openai
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
from constants import *


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(texts: list[str], model="text-embedding-ada-002") -> list[list[float]]:
    return [item["embedding"] for item in openai.Embedding.create(input=texts, model=model)["data"]]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def train_and_evaluate_embeddings_model(df, columns_for_unique_pairs=['title'], batch_size=100, RANDOM_STATE=RANDOM_STATE, TEST_SIZE=TEST_SIZE, N_ESTIMATORS=N_ESTIMATORS):
    # Assuming the user column is named 'reviewerID'

    embeddings = {}
    for column in columns_for_unique_pairs:
        column_embeddings = []
        for i in range(0, len(df[column]), batch_size):
            batch_texts = df[column].iloc[i:i+batch_size].tolist()
            column_embeddings.extend(get_embeddings(batch_texts))
        embeddings[column] = column_embeddings

    # Get embeddings for unique users
    unique_users = df['reviewerID'].unique().tolist()
    user_embeddings_dict = {}
    user_embeddings = get_embeddings(unique_users)
    for user, embedding in zip(unique_users, user_embeddings):
        user_embeddings_dict[user] = embedding

    # Create combined embeddings
    combined_embeddings = []
    for idx, row in df.iterrows():
        combined_embedding = []
        for column in columns_for_unique_pairs:
            combined_embedding += embeddings[column][idx]
        combined_embedding += user_embeddings_dict[row['reviewerID']]
        combined_embeddings.append(combined_embedding)

    X_openai = np.array(combined_embeddings)

    # Splitting the dataset
    X_train_openai, X_test_openai, y_train, y_test = train_test_split(
        X_openai, df['rating'], test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Train RandomForest on OpenAI embeddings
    rfr_openai = RandomForestRegressor(n_estimators=N_ESTIMATORS)
    rfr_openai.fit(X_train_openai, y_train)

    # Predict and evaluate
    preds_openai = rfr_openai.predict(X_test_openai)
    rmse_openai = np.sqrt(mean_squared_error(
        y_test, preds_openai))  # Calculating RMSE
    mae_openai = mean_absolute_error(y_test, preds_openai)

    print(
        f"OpenAI embedding performance: rmse={rmse_openai:.4f}, mae={mae_openai:.4f}")
