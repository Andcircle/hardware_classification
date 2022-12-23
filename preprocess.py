import pandas as pd
import os
import random
import shutil

from encoder import BertEncoder
from csv_loader import load_label_csv
from embedding_dataset import EmbeddingDataset


def preprocess(path):
    des, labels = load_label_csv(path)
    bert_encoder = BertEncoder()
    
    embeddings = bert_encoder.bert_encode(des)
    dataset = EmbeddingDataset(des, embeddings, labels)

    return dataset

    