import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    
    Parameters:
    messages_filepath (str): Path to the messages CSV file.
    categories_filepath (str): Path to the categories CSV file.
    
    Returns:
    df (pd.DataFrame): Merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged DataFrame by splitting categories and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Merged DataFrame containing messages and categories.
    
    Returns:
    df (pd.DataFrame): Cleaned DataFrame.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Use the first row to extract new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    # Drop the original categories column from df
    df = df.drop('categories', axis=1)
    
    # Concatenate df with the new categories DataFrame
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an SQLite database.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    database_filename (str): Path to SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('cleaned_data', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
