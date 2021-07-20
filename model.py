from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F

class DDEA(nn.Module):
    def __init__(self, NUM_USER, NUM_MOVIE, NUM_BOOK,  EMBED_SIZE, dropout, is_sparse=False):
        super(DDEA, self).__init__()
        self.NUM_MOVIE = NUM_MOVIE
        self.NUM_BOOK = NUM_BOOK
        self.NUM_USER = NUM_USER
        self.emb_size = EMBED_SIZE

        self.user_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()

        self.encoder_x = nn.Sequential(
            nn.Linear(self.NUM_MOVIE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
            )
        self.decoder_x = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_MOVIE)
            )
        self.encoder_y = nn.Sequential(
            nn.Linear(self.NUM_BOOK, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
            )
        self.decoder_y = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_BOOK)
            )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU
    
    def forward2(self, batch_user_1,batch_user_2, batch_user_x, batch_user_y):
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user_1 = self.user_embeddings(batch_user_1)
        h_user_2 = self.user_embeddings(batch_user_2)
        feature_x = torch.add(h_user_x, h_user_1)
        feature_y = torch.add(h_user_y, h_user_2)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)
        preds_x = self.decoder_x(z_x)
        preds_y = self.decoder_y(z_y)

        preds_x2y = self.decoder_y(z_x)
        preds_y2x = self.decoder_x(z_y)
        feature_x_r = torch.add(self.encoder_y(self.dropout(preds_x2y)),h_user_1)
        feature_y_r = torch.add(self.encoder_x(self.dropout(preds_y2x)),h_user_2)
        
        z_x_r = F.relu(feature_x_r)
        z_y_r = F.relu(feature_y_r)
        z_x_dual_loss = torch.norm(z_x-feature_x_r)
        z_y_dual_loss = torch.norm(z_y-feature_y_r)

        return preds_x, preds_y, preds_x2y, preds_y2x, feature_x, feature_y, z_x_dual_loss, z_y_dual_loss
    
    