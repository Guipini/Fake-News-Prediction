# Fake News Detection Project

## Overview

This project aims to detect fake news using machine learning techniques. It utilizes a Logistic Regression [`model`](Fake%20News%20Prediction.ipynb) trained on the [train.csv](train.csv) dataset to classify news articles as either reliable (0) or unreliable (1). The analysis includes data preprocessing, feature extraction using TF-IDF, model training, and evaluation.

## Files

-   [Fake News Prediction.ipynb](Fake%20News%20Prediction.ipynb): Jupyter Notebook containing the complete analysis and code.
-   [train.csv](train.csv): Training dataset with news articles and their labels.
-   [test.csv](test.csv): Testing dataset with news articles (without labels).
-   [submit.csv](submit.csv): Sample submission file.

## Data Preprocessing

-   Missing values in the dataset are filled with empty strings.
-   Text data from the 'title' and 'author' columns are preprocessed using stemming to reduce words to their root form.
-   Stop words (common English words) are removed to focus on meaningful content.

## Feature Extraction

-   TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is used to convert the preprocessed text into numerical features. This helps the model understand the importance of words in the articles.

## Model Training

-   A Logistic Regression [`model`](Fake%20News%20Prediction.ipynb) is trained on the processed data.
-   The dataset is split into training and testing sets to evaluate the model's performance.

## Evaluation

-   The model's accuracy is evaluated on both the training and testing datasets.
-   The accuracy score on the test data is approximately 97.91%.
-   The final accuracy, when compared against `submit.csv`, is 64.37%.

## Prediction

-   A function is defined to predict the reliability of new articles based on their title and author.
-   This function preprocesses the input text, extracts features using the trained TF-IDF vectorizer, and uses the Logistic Regression [`model`](Fake%20News%20Prediction.ipynb) to make a prediction.

## Conclusion

The Fake News Prediction pipeline demonstrates a robust approach to identifying potentially unreliable news articles. The analysis includes:

-   Data preprocessing to clean and standardize text data.
-   Feature extraction using TF-IDF to capture textual nuances.
-   Model training and evaluation using Logistic Regression, resulting in high accuracy.

Future enhancements could involve advanced feature engineering, alternative classification algorithms, or ensemble methods to improve performance and robustness.# Fake News Detection Project

## Overview

This project aims to detect fake news using machine learning techniques. It utilizes a Logistic Regression [`model`](Fake%20News%20Prediction.ipynb) trained on the [train.csv](train.csv) dataset to classify news articles as either reliable (0) or unreliable (1). The analysis includes data preprocessing, feature extraction using TF-IDF, model training, and evaluation.

## Files

-   [Fake News Prediction.ipynb](Fake%20News%20Prediction.ipynb): Jupyter Notebook containing the complete analysis and code.
-   [train.csv](train.csv): Training dataset with news articles and their labels.
-   [test.csv](test.csv): Testing dataset with news articles (without labels).
-   [submit.csv](submit.csv): Sample submission file.

## Data Preprocessing

-   Missing values in the dataset are filled with empty strings.
-   Text data from the 'title' and 'author' columns are preprocessed using stemming to reduce words to their root form.
-   Stop words (common English words) are removed to focus on meaningful content.

## Feature Extraction

-   TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is used to convert the preprocessed text into numerical features. This helps the model understand the importance of words in the articles.

## Model Training

-   A Logistic Regression [`model`](Fake%20News%20Prediction.ipynb) is trained on the processed data.
-   The dataset is split into training and testing sets to evaluate the model's performance.

## Evaluation

-   The model's accuracy is evaluated on both the training and testing datasets.
-   The accuracy score on the test data is approximately 97.91%.

## Prediction

-   A function is defined to predict the reliability of new articles based on their title and author.
-   This function preprocesses the input text, extracts features using the trained TF-IDF vectorizer, and uses the Logistic Regression [`model`](Fake%20News%20Prediction.ipynb) to make a prediction.

## Conclusion

The Fake News Prediction pipeline demonstrates a robust approach to identifying potentially unreliable news articles. The analysis includes:

-   Data preprocessing to clean and standardize text data.
-   Feature extraction using TF-IDF to capture textual nuances.
-   Model training and evaluation using Logistic Regression, resulting in high accuracy.

Future enhancements could involve advanced feature engineering, alternative classification algorithms, or ensemble methods to improve performance and robustness.