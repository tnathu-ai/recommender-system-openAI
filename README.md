# recommender-system
## Performance comparison on rating prediction

| **Methods**            | **(RMSE)** | **(MAE)** |
|------------------------|-------------------|------------------|
| MF                     | 1.1973            | 0.9461           |
| MLP                    | 1.3078            | 0.9597           |
| Paper's (zero-shot)    | 1.4059            | 1.1861           |
| Paper's (few-shot)     | 1.0751            | 0.6977           |
| Thu's OpenAI embedding using RandomForestRegressor (small Amazon)  | 1.60              | 1.14             |
| Thu's zero-shot GPT (small Amazon)          | 1.3351            | 1.2609           |
| Thu's few-shot GPT (small Amazon)           | 1.9086            | 1.0714           |
| Thu's OpenAI embedding using RandomForestRegressor (large Amazon)  | 1.60              | 1.14             |
| Thu's zero-shot GPT (large Amazon)          | 1.1344            | 1.0118           |
| Thu's few-shot GPT (large Amazon)           | 1.9086            | 1.0714           |

**Paper's Few-shot:**
RMSE: 1.0751 - This is the lowest RMSE among the methods, indicating that the few-shot approach from the paper is the most accurate in terms of squared differences.

**Thu's Few-shot GPT:**
RMSE: 1.9086 - This is the highest RMSE among the methods, indicating that this approach has the largest deviation in terms of squared differences.
![image](https://github.com/tnathu-ai/recommender-system/assets/72063833/45f92fdc-32f4-425c-bcd4-dfdb331ca5f4)

## Repository Structure

- `LICENSE`: This file contains the licensing details, which describes how the project can be used or shared.
- `README.md`: This file contains a general description of the project, setup instructions, examples of use, and other important details.
