# recommender-system
## Performance comparison on rating prediction

| **Methods**            | **Beauty (RMSE)** | **Beauty (MAE)** |
|------------------------|-------------------|------------------|
| MF                     | 1.1973            | 0.9461           |
| MLP                    | 1.3078            | 0.9597           |
| Paper's (zero-shot)    | 1.4059            | 1.1861           |
| Paper's (few-shot)     | 1.0751            | 0.6977           |
| Thu's OpenAI embedding using RandomForestRegressor      | 1.59              | 1.13             |
| Thu's zero-shot GPT          | 1.3351            | 1.2609           |
| Thu's few-shot GPT           | 1.9086            | 1.0714           |


**Paper's Few-shot:**
RMSE: 1.0751 - This is the lowest RMSE among the methods, indicating that the few-shot approach from the paper is the most accurate in terms of squared differences.

**Thu's Few-shot GPT:**
RMSE: 1.9086 - This is the highest RMSE among the methods, indicating that this approach has the largest deviation in terms of squared differences.
![image](https://github.com/tnathu-ai/recommender-system/assets/72063833/45f92fdc-32f4-425c-bcd4-dfdb331ca5f4)

## Repository Structure

- `LICENSE`: This file contains the licensing details, which describes how the project can be used or shared.
- `README.md`: This file contains a general description of the project, setup instructions, examples of use, and other important details.

- `SVD`: This is a directory that contains everything related to the SVD project.

  - `data`: This directory contains a dataset `ml-latest-small` from MovieLens.
    - `ml-latest-small`: It includes several CSV files like `movies.csv`, `ratings.csv`, `links.csv`, `tags.csv`, and a `README.txt` explaining the dataset. There's also a PDF file `ml-latest-small-README.pdf` for additional information about the dataset.
  - `notebook`: This directory includes a Jupyter notebook `SVD.ipynb` which contains the code for the SVD implementation.

- `resources`: This directory contains additional resources related to SVD.

For further details about the implementation, please refer to the Jupyter notebook under the `SVD/notebook` directory.
