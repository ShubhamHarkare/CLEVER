import pandas as pd
from datasets import load_dataset,train_test_split
import os
from typing import List,Any

DATASET_NAME = 'lmsys/lmsys-chat-1m'
# DATASET_PATH = "../data/processed/cleaned_data.parquet"
TRAIN_DATASET_PATH = '../data/processed/train_data'
TEST_DATA = '../data/processed/test_data'
HOLDOUT_DATA = '../data/processed/holdout_data'
SAMPLE_SIZE = 10000
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clever_dev_set")


def prepareData():
    '''
    Prepares the data for training.
    '''
    try:
       data = load_dataset(DATASET_NAME)
       print(f'Shape of the data : {len(data)}')
    except Exception as e:
       print(f'Error loading the dataset: {e}')
       return
    # Step-1 Filetring the data based on langauage
    data_english = data.filter(lambda example: example['language'] == 'English')
    print(f'Shape after language filtering: {len(data_english)}')
    
    # Step-2 Checking if the first content is by user and also within our agreed limits
    filtered_data = data_english.filter(lambda row: 20 < len(row['conversation'][0]['content']) < 2000) 
    print(f'Shape after content length filtering: {len(filtered_data)}')
   
   # Step-3: Saving the data for further use
    print('Saving the data...')
    os.makedirs(os.path.dirname(SAVE_PATH),exist_ok=True)
    filtered_data.save_to_disk(SAVE_PATH)
    print("Data saved successfully")  

    
    
    

if __name__ == "__main__":
    prepareData()