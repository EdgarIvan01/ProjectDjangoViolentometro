# Django Project: Classifying Violent Words Using Natural Language Processing (NLP)

## Project Description

This project is a web application developed in Django that processes natural language text to classify words as "violent" or "non-violent". It uses natural language processing (NLP) techniques to cleanse the text and a Naive Bayes (NB) model trained on a pre-labeled dataset to predict the classification of new words.

## Project Structure

- **Django App**: The Django app serves as the backend to handle HTTP requests and perform text processing.
- **Natural Language Processing (NLP)**: Implemented in Python, this module cleans and processes the text to remove noise (such as punctuation, numbers, etc.) and convert the text into a format suitable for analysis.
- **Naive Bayes (NB) Model**: An NB model is trained using a CSV file containing words labeled as "violent" or "non-violent". This model is used to predict the classification of new words.
- **Model Training**: The model is trained using labeled data. This data is stored in a CSV file which is uploaded and processed in the training stage.
- **Prediction**: Once trained, the NB model is used to classify new words submitted through the web interface.

## Main Flows

1. **Data Loading and Cleaning**: Labeled data is loaded from a CSV file and cleaned using NLP techniques such as stop word removal, tokenization, and lemmatization.

2. **Model Training**: The Naive Bayes model is trained using the cleaned and labeled data. A trained model is stored for future use.

3. **Web Interface**: Users can enter new words or phrases through the Django web interface. The system cleans and processes the entered text, and then uses the NB model to predict whether the words are "violent" or "non-violent."

4. **Results Display**: The ranking results are displayed in the web interface.
