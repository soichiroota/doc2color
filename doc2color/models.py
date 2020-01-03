from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer, BertModel, BertForMaskedLM


class BertModelWithTokenizer():
    def __init__(self, bert_path, use_cuda=False):
        self.model = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.use_cuda = use_cuda
        self.max_position_embeddings = self.model.config.max_position_embeddings

    def get_sentence_embedding(self, text, pooling_strategy="REDUCE_MEAN"):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenized_text[:self.max_position_embeddings-2] + ["[SEP]"])
        tokens_tensor = torch.tensor(indexed_tokens).reshape(1, -1)

        if self.use_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            self.model.to('cuda')

        self.model.eval()
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor)

        embedding = encoded_layers[0].cpu().numpy()
        if pooling_strategy == "REDUCE_MEAN":
            return np.mean(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MAX":
            return np.max(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]
        elif pooling_strategy == "CLS_TOKEN":
            return embedding[0]
        else:
            raise ValueError("specify valid pooling_strategy: {REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN}")

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        #self.fc6 = nn.Linear(32, 16)
        #self.fc7 = nn.Linear(16, 3)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(64)
        #self.batch_norm5 = nn.BatchNorm1d(32)
        #self.batch_norm6 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.batch_norm3(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.batch_norm4(x)
        x = self.fc5(x)
        #x = F.relu(x)
        #x = self.batch_norm5(x)
        #x = self.fc6(x)
        #x = F.relu(x)
        #x = self.batch_norm6(x)
        #x = self.fc7(x)
        output = torch.sigmoid(x)
        return output
