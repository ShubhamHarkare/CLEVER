import pandas as pd
from datasets import load_dataset
import os
from typing import List,Any

DATASET_NAME = 'lmsys/lmsys-chat-1m'
# DATASET_PATH = "../data/processed/cleaned_data.parquet"
TRAIN_DATASET_PATH = '../data/processed/train_data'
TEST_DATA_PATH = '../data/processed/test_data'
HOLDOUT_DATA_PATH = '../data/processed/holdout_data'
SAMPLE_SIZE = 10000
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clever_dev_set")


def prepareData():
    '''
    Prepares the data for training.
    '''
    try:
       data = load_dataset(DATASET_NAME,split ='all')
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
    # Step-3: train-test and validation split
    train_test = filtered_data.train_test_split(test_size = 0.2,seed = 42)
    train_data,test_data = train_test['train'],train_test['test']
    train_val = train_data.train_test_split(test_size = 0.2,seed = 42)
    train_data,val_data = train_val['train'],train_val['test']
    
    print(f'Training data size: {train_data.shape}')
    print(f'Testing data size: {test_data.shape}')
    print(f'Validation data size: {val_data.shape}')
    
    
    
    # Step-4: Saving the data for further use
    print('Saving the data...')
    # os.makedirs(os.path.dirname(SAVE_PATH),exist_ok=True)
    os.makedirs(os.path.dirname(TRAIN_DATASET_PATH),exist_ok=True)
    os.makedirs(os.path.dirname(TEST_DATA_PATH),exist_ok=True)
    os.makedirs(os.path.dirname(HOLDOUT_DATA_PATH),exist_ok=True)
    train_data.save_to_disk(TRAIN_DATASET_PATH)
    test_data.save_to_disk(TEST_DATA_PATH)
    val_data.save_to_disk(HOLDOUT_DATA_PATH)
    # filtered_data.save_to_disk(SAVE_PATH)
    print("Data saved successfully")  

    
    
    

if __name__ == "__main__":
    prepareData()