import os

import torch
import numpy as np
import logging
import time
import random

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch import nn

from classifier import Classifier
from preprocess import preprocess

def test_model(test_dataloader, model):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for b, data_batch in enumerate(test_dataloader):
        #send data to the device
        embeddings = data_batch['embedding'].to(device)
        labels = data_batch['label'].to(device)
        des = data_batch['des']

        outputs = model(embeddings)
        _, predicted = torch.max(outputs.data, 1)
                
        for d, p, l in zip(des, predicted.detach().cpu().numpy(), labels):
            print("description: {}, predicted: {}, label: {}".format(d, p, l))
        
        break
        
    model.train()

def train_model(train_dataloader, test_dataloader, model):

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=3.0e-4, momentum=0.9)

    epoches = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(epoches):
        t1 = time.time()
        
        for i in range(1000): # this for loop should be remove, we added here because the dataset is too small
            for b, data_batch in enumerate(train_dataloader):
                #send data to the device
                embeddings = data_batch['embedding'].to(device)
                labels = data_batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        print("Epoch {}, Loss: {}".format(epoch, loss))
        results = test_model(test_dataloader, model)
             

def train():
        
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Classifier()
        
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    dataset = preprocess('training_data.csv')
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    train_model(dataloader, dataloader, model)

    
if __name__ == "__main__":
    train()

