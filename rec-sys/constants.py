import os

STOP_AFTER_N_ATTEMPTS = 7

# Data Parameters
# Random State
RANDOM_STATE = 2002
# Number of Samples
NUM_SAMPLES = 100

# train-test split
TEST_SIZE = 0.2

# Model Parameters
# batch size
BATCH_SIZE = 10
# n_estimators: the number of trees in the forest of the model
N_ESTIMATORS = 10
# number of examples to show
NUM_EXAMPLES = 5


# OpenAI API Parameters
# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI GPT Model parameters
GPT_MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0
# Embedding Embedding
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002
MAX_TOKENS = 8000  # the maximum for text-embedding-ada-002 is 8191
MAX_TOKENS_CHAT_GPT = 16000  # the maximum for chat is 2048

# Parameters for pausing
PAUSE_EVERY_N_USERS = 10
SLEEP_TIME = 60  # Sleep for 60 seconds


# Columns related to the item side can be used for the prediction and training
ITEM_SIDE = [
    "asin",
    "title",
    "feature",
    "description",
    "price",
    "brand",
    "category",
    "tech1",
    "tech2",
    "also_buy",
    "also_view",
    "details",
    "main_cat",
    "similar_item",
    "date",
    "rank"
]

# Columns related to the interaction side can be used for training but can't be used for the prediction
INTERACTION_SIDE = [
    "reviewText",
    "rating",
    "summary",
    "unixReviewTime",
    "reviewTime",
    "vote",
    "style",
]

# Columns related to the user side can be used for training but can't be used for the prediction
USER_SIDE = [
    "reviewerID",
    "reviewerName",
    "verified"
]
