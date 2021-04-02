import torch
import torch.nn as nn
from torch.nn import parameter
from models import F3Net
import torch.nn.functional as F
import numpy as np
import os


def initModel(mod, gpu_ids):
    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod

class Trainer(): 
    def __init__(self, gpu_ids, mode, pretrained_path):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.model = F3Net(mode=mode, device=self.device)
        self.model = initModel(self.model, gpu_ids)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=0.0002, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                                         lr=0.002, momentum=0.9, weight_decay=0)

    def set_input(self, input, label):
        self.input = input.to(self.device)
        self.label = label.to(self.device)

    def forward(self, x):
        fea, out = self.model(x)
        del fea
        return out
    
    def optimize_weight(self):
        stu_fea, stu_cla = self.model(self.input)

        self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label) # classify loss
        self.loss = self.loss_cla

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
