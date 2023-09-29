## Performance comparison on rating prediction

| **Methods**            | **Beauty (RMSE)** | **Beauty (MAE)** |
|------------------------|-------------------|------------------|
| MF                     | 1.1973            | 0.9461           |
| MLP                    | 1.3078            | 0.9597           |
| Paper's (zero-shot)    | 1.4059            | 1.1861           |
| Paper's (few-shot)     | 1.0751            | 0.6977           |
| Thu's OpenAI embedding       | 1.59              | 1.13             |
| Thu's zero-shot GPT          | 1.3351            | 1.2609           |
| Thu's few-shot GPT           | 1.9086            | 1.0714           |


**Paper's Few-shot:**
RMSE: 1.0751 - This is the lowest RMSE among the methods, indicating that the few-shot approach from the paper is the most accurate in terms of squared differences.

**Thu's Few-shot GPT:**
RMSE: 1.9086 - This is the highest RMSE among the methods, indicating that this approach has the largest deviation in terms of squared differences.