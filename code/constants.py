import os

# =============================================================================
# General Constants
# =============================================================================
STOP_AFTER_N_ATTEMPTS = 7  # Maximum number of retry attempts for a function

# =============================================================================
# Data Parameters
# =============================================================================
RANDOM_STATE = 2002  # Seed value for random number generation
NUM_SAMPLES = 100    # Number of samples to consider in operations
TEST_SIZE = 0.2      # Proportion of the dataset to include in the test split

# =============================================================================
# Machine Learning Model Parameters
# =============================================================================
BATCH_SIZE = 10      # Batch size for model training
N_ESTIMATORS = 10    # Number of trees in the forest for ensemble models
NUM_EXAMPLES = 5     # Number of examples to show for demonstrations

# =============================================================================
# OpenAI API Parameters
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # API Key for OpenAI

# GPT Model Parameters
GPT_MODEL_NAME = "gpt-3.5-turbo-1106"  # Model name for GPT
TEMPERATURE = 0                   # Sampling temperature for model response generation

# Embedding Model Parameters
EMBEDDING_MODEL = "text-embedding-ada-002"  # Model name for text embedding
EMBEDDING_ENCODING = "cl100k_base"          # Encoding for text-embedding-ada-002
MAX_TOKENS = 8000                           # Maximum tokens for embedding (limit is 8191)

# Chat GPT Model Parameters
MAX_TOKENS_CHAT_GPT = 4000  # Maximum tokens for chat responses, considering the response tokens

# =============================================================================
# Parameters for Processing Control
# =============================================================================
PAUSE_EVERY_N_USERS = 10  # Pause frequency in terms of number of users processed
SLEEP_TIME = 60        # Duration to pause in seconds (e.g., 60 seconds)

# =============================================================================
# Dataset Specific Constants for Amazon Beauty dataset
# =============================================================================
# Item-side attributes useful for prediction and training
ITEM_SIDE = [
    "asin", "title", "feature", "description", "price", "brand", "category",
    "tech1", "tech2", "also_buy", "also_view", "details", "main_cat",
    "similar_item", "date", "rank"
]

# Interaction-side attributes for training (not for prediction)
INTERACTION_SIDE = [
    "reviewText", "rating", "summary", "unixReviewTime", "reviewTime", "vote",
    "style"
]

# User-side attributes for training (not for prediction)
USER_SIDE = [
    "reviewerID", "reviewerName", "verified"
]

# =============================================================================
# Evaluation Metrics Parameters
# =============================================================================
# confidence level for the confidence interval
CONFIDENCE_LEVEL = 0.95
# Bootstrap Resampling: Repeatedly sample set of (actual, predicted) ratings with replacement, typically thousands of times (e.g., 10,000 bootstrap samples).
BOOSTRAP_RESAMPLING_ITERATIONS = 10000
# confidence_multiplier
CONFIDENCE_MULTIPLIER = 1.96

# =============================================================================
# OpenAI API chat completion response parameters
# =============================================================================
AMAZON_CONTENT_SYSTEM = "Amazon Beauty products critic"
MOVIELENS_CONTENT_SYSTEM = "MovieLens movies critic"