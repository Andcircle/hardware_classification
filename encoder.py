import torch
import logging
from torch.utils.data import DataLoader
from transformers import DistilBertModel
from token_dataset import TokenDataset


class BertEncoder:
    def __init__(self):
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = DistilBertModel.from_pretrained("../distilbert-base-uncased")
        self.bert_model.eval()
        self.bert_model = self.bert_model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)


    def _bert_encode(self, dataloader):
        embedding_list = []
        
        for n,b in enumerate(dataloader):
            b={k: v.squeeze(0).to(self.device) for k,v in b.items()}
            out = self.bert_model(**b)
            # we only want the hidden_states
            last_hidden_states = out[0]

            embeddings = torch.mean(last_hidden_states, dim=1)
            embeddings = list(torch.split(embeddings.detach().cpu(), 1))
            embeddings = [embedding.squeeze().numpy() for embedding in embeddings]
            embedding_list += embeddings

        return embedding_list
    
    
    def bert_encode(self, des):
        token_dataset = TokenDataset(des)
        token_loader = DataLoader(token_dataset, batch_size=1, num_workers=3)
        embeddings =  self._bert_encode(token_loader)
        
        return embeddings