# constants.py
import os

# Random State
RANDOM_STATE = 2002
# Number of Samples
NUM_SAMPLES = 100
# batch size
BATCH_SIZE = 10
# n_estimators: the number of trees in the forest of the model
N_ESTIMATORS = 10


# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI GPT Model parameters
GPT_MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0
