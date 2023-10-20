# Recommender System 

A collection of performance comparisons on rating prediction for various recommender system methods. 

**Latest Update**: Added ReviewText as an additional feature.

<details><summary><b> Data Overview</b></summary>
<p>
Amazon Dataset Description
The Amazon dataset used in this project is divided into two sets: Small Amazon and Large Amazon.

1. **Small Amazon Dataset:**

This dataset is a subset of the main dataset where the 'reviewerID' column is used to filter out users.
Only users with more than 5 ratings are considered.
From these users, records of only 5 unique users are randomly selected using a specific seed.
Dataset Statistics:
+ Number of unique users: 5
+ Number of unique products: 23
+ Number of unique ratings: 4
+ Unique rating values: [5.0, 2.0, 3.0, 4.0]

2. **Large Amazon Dataset:**

This dataset includes all data that has a 'reviewerID'.
Only users with 5 or more ratings are considered.
Dataset Statistics:
+ Number of unique users: 1608
+ Number of unique products: 1879
+ Number of unique ratings: 5
+ Unique rating values: [1.0, 5.0, 4.0, 2.0, 3.0]

</p>
</details>

<details><summary><b> OpenAI API Performance Comparison on Rating Prediction</b></summary>
<p>

| **Methods**                                          | **Dataset**           | **Feature(s)** | **Model Name**        | **Parameters**                                    | **RMSE** | **(MAE)** | **Wall Time** |
|------------------------------------------------------|-----------------------|--------------|-----------------------|---------------------------------------------------|------------|-----------|----------------|
| MF [1]                                               | Unknown               | title        | -                     | -                                                 | 1.1973     | 0.9461    | -              |
| MLP [2]                                              | Unknown               | title        | -                     | -                                                 | 1.3078     | 0.9597    | -              |
| Paper's (zero-shot) [3]                              | Unknown Amazon        | title        | GPT-3.5-turbo                     | -                                                 | 1.4059     | 1.1861    | -              |
| Paper's (few-shot) [3]                               | Unknown Amazon        | title        | GPT-3.5-turbo                     | -                                                 | 1.0751     | 0.6977    | -              |
| Thu's OpenAI embedding                               | Small Amazon          | title        | RandomForestRegressor | BATCH_SIZE=10, N_ESTIMATORS=10, MAX_TOKENS=8000    | 1.6036       | 1.1429      | 47.9 ms        |
| Thu's zero-shot GPT                                  | Small Amazon          | title        | GPT-3.5-turbo         | TEMPERATURE=0, MAX_TOKENS=8000                     | 1.3351     | 1.2609    | 13.6 s              |
| Thu's few-shot GPT                                   | Small Amazon          | title        | GPT-3.5-turbo         | TEMPERATURE=0, MAX_TOKENS=8000                     | 1.9086     | 1.0714    | 16.4 s              |
| Thu's OpenAI embedding                               | Large Amazon          | title        | RandomForestRegressor | BATCH_SIZE=10, N_ESTIMATORS=10, MAX_TOKENS=8000    | 0.6240       | 0.3107      | 1h 25min 35s              |
| Thu's zero-shot GPT                                  | Large Amazon          | title        | GPT-3.5-turbo         | TEMPERATURE=0, MAX_TOKENS=8000                     | 1.1344     | 1.0118    | 13h 14min 39s              |
| Thu's few-shot GPT                                   | Large Amazon          | title        | GPT-3.5-turbo         | TEMPERATURE=0, MAX_TOKENS=8000                     | 0.7185     | 0.3259    | 9h 36min 7s             | 
| Thu's few-shot GPT (1 test/user)                                    | Large Amazon          | title        | GPT-3.5-turbo         | TEMPERATURE=0, MAX_TOKENS=8000                     | 0.6445     | 0.2226    | 15h 37s              | 
| Thu's zero-shot GPT                                  | Small Amazon            | title, reviewText        | GPT-3.5-turbo         | TEMPERATURE=0, MAX_TOKENS=8000                     | 1.3758     | 1.0118    | 12min 21s              |
| Thu's few-shot GPT                                   | Small Amazon           | title, reviewText        | GPT-3.5-turbo         | TEMPERATURE=0, MAX_TOKENS=8000                     | 1.9457     | 0.9286    | 10min 30s              | 
| Thu's few-shot GPT (1 test/user)                                  | Small Amazon           | title, reviewText        | GPT-3.5-turbo         | TEMPERATURE=0, MAX_TOKENS=8000                     | 0.6325     | 0.4   | 9.59 s             | 

**References:**

[1] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009), 30–37.

[2] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems. 7–10.

[3] [https://arxiv.org/pdf/2304.10149.pdf](https://arxiv.org/pdf/2304.10149.pdf)


</p>
</details>

<details><summary><b>SVD in Recommender System</b></summary>
<p>

![image](https://github.com/tnathu-ai/recommender-system/assets/72063833/45f92fdc-32f4-425c-bcd4-dfdb331ca5f4)

</p>
</details>

## non-LaTeX example from training_data.jsonl:

  ```
  {
    "prompt": "Title: Charming Silver Colored Earring / Ear Cuff / Clip In Snake / Spiral Shape By VAGA", 
    "completion": "5.0"
  }
  ```

## Repository Structure

- `LICENSE`: Licensing details.
- `README.md`: Description of the project, setup instructions, and other details.

Please check the repository regularly for updates and performance improvements.

