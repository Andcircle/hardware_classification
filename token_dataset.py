import torch
from torch.utils.data import Dataset

import logging

from transformers import DistilBertTokenizer


class TokenDataset(Dataset):

    def __init__(self, des):

        self.des = des
        self.tokenizer = DistilBertTokenizer.from_pretrained("../distilbert-base-uncased")
        batch_size = 256
        
        self.des_batches = []
        for s in range(0, len(des), batch_size):
            des_batch=des[s:min(s+batch_size,len(des))]
            self.des_batches.append(des_batch)


    def __len__(self):
        return len(self.des_batches)


    def __getitem__(self, idx):
        return self.tokenizer(self.des_batches[idx], padding='max_length', truncation=True, max_length=200, return_tensors='pt')