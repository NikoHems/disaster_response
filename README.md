# Disaster Response Pipeline Project

This project is part of the Udacity Data Scientist Nanodegree Program. The goal of this project is to build a web application that can help emergency workers classify disaster response messages into various categories. This allows the appropriate emergency services to be allocated efficiently and effectively.

## Table of Contents
- [Project Overview](#project-overview)
- [Instructions](#instructions)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Web App](#running-the-web-app)
- [Acknowledgements](#acknowledgements)

## Project Overview

The project includes the following steps:
1. **ETL Pipeline**: Extract, transform, and load data from disaster messages and categories datasets, clean and store in a SQLite database.
2. **Machine Learning Pipeline**: Train a machine learning model to classify disaster messages, evaluate the model, and save the trained model as a pickle file.
3. **Web Application**: A Flask web app to display visualizations of the data and allow users to input messages and get classification results.

## Instructions

Run the following commands in the project's root directory to set up your database and model.

1. **ETL Pipeline**: Clean data and store in the database.
    ```sh
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```

2. **ML Pipeline**: Train classifier and save the model.
    ```sh
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

3. **Run the Web App**:
    Navigate to the app's directory and run the following command to start the web app.
    ```sh
    python run.py
    ```

4. **Open the Web App**:
    Go to [http://0.0.0.0:3001/](http://0.0.0.0:3001/) in your browser.

## File Descriptions

- **app**
  - `run.py`: Flask file that runs the web application.
  - `templates/`: HTML templates for the web app.

- **data**
  - `disaster_categories.csv`: Dataset containing disaster categories.
  - `disaster_messages.csv`: Dataset containing disaster messages.
  - `DisasterResponse.db`: SQLite database to store the cleaned data.
  - `process_data.py`: ETL pipeline script to process data and save it to a SQLite database.

- **models**
  - `train_classifier.py`: Machine learning pipeline script to train the model and save it as a pickle file.
  - `classifier.pkl`: Saved model after training.

## Dependencies

- Python 3.x
- pandas
- numpy
- sqlalchemy
- scikit-learn
- nltk
- flask
- plotly

Install the required libraries using:
```sh
pip install -r requirements.txt
