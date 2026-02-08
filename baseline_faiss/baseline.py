# This file is to figure out the baseline model ising FAISS index Flat L2

import numpy as np
import faiss
import time
from datasets import load_from_disk
import pandas as pd
import pyarrow.parquet as pq
import glob
import os
from sentence_transformers import SentenceTransformer

DATA_PATH = "../data/processed/clever_dev_set"
MODEL_NAME = 'all-MiniLM-L6-v2'
TEST_SIZE = 1000 # We will test on a 1000 queries
K = 10 # We want the 10 most closest vectors

def runBaseline():
    '''
    This function will implement the faissIndexL2 indexing 
    to find out the best vector for the baseline model
    '''
    # Step-1 loading the data
    print(f'Loading the data from {DATA_PATH}')
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file {DATA_PATH} not found.")
        return
        
    
    dataset = load_from_disk(DATA_PATH)
    print(dataset.shape)
    
if __name__ == "__main__":
    runBaseline()
    