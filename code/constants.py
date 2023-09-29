# constants.py
import os

# Random State
RANDOM_STATE = 2002

# Number of Samples
NUM_SAMPLES = 100

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NUM_FEATURES, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE = 100, 2, 10, 32, 0.01
