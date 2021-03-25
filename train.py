import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import sys
import time
import torch
import torch.nn

from utils import evaluate, get_dataset, FFDataset
from trainer import Trainer
import numpy as np
import random

# config
dataset_path = '/data/yike/FF++_std_c40_300frames/'
pretrained_path = 'pretrained/xception-b5690688.pth'
batch_size = 12
gpu_ids = [*range(osenvs)]
max_epoch = 2
loss_freq = 40
mode = 'FAD' # ['FAD', 'LFS', 'Both', 'Mix']


if __name__ == '__main__':
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'), size=299, frame_num=300)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8)
    
    len_dataloader = dataloader_real.__len__()

    dataset_img, total_len =  get_dataset(name='train', size=299, root=dataset_path, frame_num=300)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8
    )
    

    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    epoch = 0
    
    while epoch < max_epoch:

        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)
        
        print(f'No {epoch}')
        i = 0

        while i < len_dataloader:
            
            i += 1
            model.total_steps += 1

            try:
                data_real = real_iter.next()
                data_fake = fake_iter.next()
            except StopIteration:
                break
            # -------------------------------------------------
            
            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz = data_real.shape[0]
            
            data = torch.cat([data_real,data_fake],dim=0)
            label = torch.cat([torch.zeros(bz).unsqueeze(dim=0),torch.ones(bz).unsqueeze(dim=0)],dim=1).squeeze(dim=0)

            model.set_input(data,label)
            loss = model.optimize_weight()

            if model.total_steps % loss_freq == 0:
                print(f'loss: {loss} at step: {model.total_steps}')

            if i % int(len_dataloader / 10) == 0:
                model.model.eval()
                auc, r_acc, f_acc = evaluate(model, data_path, mode='valid')
                print(f'(Val @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
                model.model.train()
        epoch = epoch + 1

    auc, r_acc, f_acc = evaluate(model, data_path, mode='test')
    print(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
