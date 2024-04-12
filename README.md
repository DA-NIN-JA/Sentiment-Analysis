# Restaurant Reviews Sentiment Analysis

## Introduction
This project aims to perform sentiment analysis on restaurant reviews using Natural Language Processing (NLP) techniques. It utilizes a dataset of restaurant reviews to train a machine learning model to classify reviews as positive or negative.

## Dataset
The dataset consists of restaurant reviews stored in the file "Restaurant_Reviews.tsv". Each review is labeled as either positive or negative. The data is preprocessed and cleaned to remove any unnecessary characters and stopwords.

## Data Preprocessing
The text data is cleaned by removing non-alphabetic characters, converting text to lowercase, and applying stemming to reduce words to their root form. Stopwords are removed to focus on meaningful words for sentiment analysis.

## Model Training
A Support Vector Machine (SVM) classifier with a linear kernel is trained on the preprocessed text data. The CountVectorizer is used to convert text data into numerical features. The dataset is split into training and test sets for model evaluation.

## Model Evaluation
The trained SVM model is evaluated on the test set using metrics such as accuracy and confusion matrix. The confusion matrix provides insights into the model's performance in correctly classifying positive and negative reviews.

## Usage
To use the trained model for sentiment analysis on new reviews, simply input a review text, and the model will predict whether it is positive or negative. The input review undergoes the same preprocessing steps as the training data before being fed into the model.

## Example
An example of using the trained model to predict the sentiment of a new review is provided. Users can input their review text, and the model will output whether it is positive or negative.

## Dependencies
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- NLTK

## How to Run
1. Ensure all dependencies are installed.
2. Import the dataset.
3. Preprocess the text data.
4. Train the SVM model.
5. Evaluate the model.
6. Input a new review to predict its sentiment.

## Conclusion
This project demonstrates the application of NLP techniques for sentiment analysis on restaurant reviews. The trained model can be used to analyze customer feedback and make data-driven decisions to improve restaurant services.
