import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):

    def __init__(self, des, embeddings, labels):

        self.des = des
        self.embeddings = embeddings
        self.labels = labels
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __len__(self):
        return len(self.des)


    def __getitem__(self, idx):
        return {"des":self.des[idx], "embedding":  self.embeddings[idx], "label":self.labels[idx]}
