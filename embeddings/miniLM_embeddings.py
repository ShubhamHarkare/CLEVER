# This file is responsible for embedding text using the miniLM model.
import numpy as np
from numpy.random.mtrand import normal
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from typing import List
from tqdm import tqdm
import torch
import os
# TODO: Using this model
DATA_PATH = '../data/processed/'
OUTPUT_PATH = '../data/embeddings/'
class GetEmbeddings():
    '''
    This class is responsible for embedding text using the miniLM model.
    '''
    def __init__(self,data_location:str,model_name:str,device:str):
        self.texts :List[str] = []
        self.data_location  = data_location
        # Below data helps in loading the embedding model
        try:
            print(f'Loading the model {model_name} on {device}')
            self.model = SentenceTransformer(model_name,device = device)
            print('Successfully loaded the model')
        except Exception as e:
            print(f'Error Loading the model: {e}')
            raise e
            
        
        try:
            full_path = os.path.join(DATA_PATH, data_location)
            self.data = load_from_disk(full_path)
            print(f'Successfully loaded the data : {self.data.shape}')
        except Exception as e:
            print(f'Error loading the data: {e}')
            raise e
            
    def _processConversation(self,conversation_list : List[str]) -> str:
        '''
        This function will process the conversation list and return a string
        '''
        return ' '.join([f'{msg['role']}: {msg['content']}' for msg in conversation_list])

    def embedData(self) -> None:
        '''
        This function will embed the data using the model and the device that you have 
        used to embed the model
        '''
        print("Preparing the data for embedding")
        
        self.texts = [
            self._processConversation(row['conversation']) for row in tqdm(self.data, desc = 'Extracting conversations')
        ]
        print(f'Total converstions : {len(self.texts)}')
        embeddings = self.model.encode(
            self.texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy = True,
            normalize_embeddings = True
        )
        
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        save_path = os.path.join(OUTPUT_PATH, f'{self.data_location}.npy')
        np.save(save_path, embeddings)
                
        print(f'Successfully saved embeddings to {save_path}')
        print(f'Final Embedding Shape: {embeddings.shape}')

            

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE IN USE: {device}")
    embeddings = GetEmbeddings(data_location='train_data', model_name='all-MiniLM-L6-v2', device=device)
    embeddings.embedData()