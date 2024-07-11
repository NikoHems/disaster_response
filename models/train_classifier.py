import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score, accuracy_score
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Parameters:
    database_filepath (str): Path to SQLite database file.
    
    Returns:
    X (pd.Series): Messages.
    Y (pd.DataFrame): Categories.
    category_names (list): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('cleaned_data', engine)  
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    
    # Convert all target columns to numeric values
    Y = Y.apply(pd.to_numeric, errors='coerce')
    
    # Ensure no columns contain NaN values
    Y = Y.fillna(0).astype(int)
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and preprocess text data.
    
    Parameters:
    text (str): The text to be tokenized and processed.
    
    Returns:
    list: A list of tokens (words) after processing.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def build_model():
    """
    Build a machine learning pipeline and perform grid search.
    
    Returns:
    model (GridSearchCV): Grid search model object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 3]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2, verbose=2)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print out the classification report for each category.
    
    Parameters:
    model: Trained model.
    X_test (pd.Series): Test messages.
    Y_test (pd.DataFrame): Test categories.
    category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print(f"Classification report for '{category_names[i]}':")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
    
    accuracy = (Y_pred == Y_test).mean().mean()
    print(f'\nOverall accuracy: {accuracy}')

def save_model(model, model_filepath):
    """
    Save the model to a pickle file.
    
    Parameters:
    model: Trained model.
    model_filepath (str): Path to save the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filepath}")

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
