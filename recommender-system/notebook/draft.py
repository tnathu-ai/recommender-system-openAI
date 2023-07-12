% % time

# Create mappings for userIds and movieIds to contiguous indices
user_mapping = {user_id: i for i,
                user_id in enumerate(M400_df['userId'].unique())}
movie_mapping = {movie_id: i for i,
                 movie_id in enumerate(M400_df['movieId'].unique())}

# Create reverse mappings for later use
reverse_user_mapping = {i: user_id for user_id, i in user_mapping.items()}
reverse_movie_mapping = {i: movie_id for movie_id, i in movie_mapping.items()}

# Apply the mappings to the dataframes
M400_df['userId'] = M400_df['userId'].map(user_mapping)
M400_df['movieId'] = M400_df['movieId'].map(movie_mapping)

test_df['userId'] = test_df['userId'].map(user_mapping)
test_df['movieId'] = test_df['movieId'].map(movie_mapping)

# Drop rows with NaN userId or movieId
test_df.dropna(subset=['userId', 'movieId'], inplace=True)

# Convert userId and movieId to integer
test_df['userId'] = test_df['userId'].astype(int)
test_df['movieId'] = test_df['movieId'].astype(int)


n_users = M400_df['userId'].nunique()
n_items = M400_df['movieId'].nunique()

train_matrix = df_to_matrix(M400_df, n_users, n_items)
test_matrix = df_to_matrix(test_df, n_users, n_items)

knn_cf = KNN_CF(n_users, n_items, k=3)

# Fit the model to the M100 data
knn_cf.fit(train_matrix)

# Predict ratings for the Test set and evaluate
user_based_predictions = knn_cf.predict(test_matrix, mode='user')
test_predictions = user_based_predictions[test_matrix.nonzero()]
actual_ratings = test_matrix[test_matrix.nonzero()]

knn_predictions = [(uid, iid, true_r, est) for uid, iid, true_r, est in zip(
    test_df['userId'], test_df['movieId'], actual_ratings, test_predictions)]
# Compute metrics for the KNN model
knn_metrics_M100 = evaluate(knn_predictions)

# create a dataframe to concatenate the results
knn_results_M50 = pd.DataFrame(knn_metrics_M100, index=[0])
knn_results_M50
