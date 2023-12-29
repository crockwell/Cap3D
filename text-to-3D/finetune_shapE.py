# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Based on https://github.com/openai/shap-e
#
# This code is licensed under the MIT License.
# ==============================================================================


import torch
import torch.optim as optim

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.models.configs import model_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from IPython import embed

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import argparse

import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import csv
import time
import random
import numpy as np
from datetime import datetime


def setup_ddp(gpu, args):
    dist.init_process_group(                                   
        backend='nccl',      # backend='gloo',#                                    
        init_method='env://',     
        world_size=args.world_size,                              
        rank=gpu)

    torch.cuda.set_device(gpu)

class shapE_train_dataset(Dataset):
    def __init__(self, latent_code_path):
        self.captions = pd.read_csv('./example_material/Cap3D_automated_Objaverse.csv', header=None)
        self.valid_uid = list(pickle.load(open('./example_material/training_set.pkl','rb')))
        self.final_uid = self.valid_uid
        self.n2idx = {}
        for i in range(len(self.captions)):
            self.n2idx[self.captions[0][i]] = i
        self.latent_code_path = latent_code_path        

    def __len__(self):
        return len(self.final_uid)

    def __getitem__(self, i):
        idx = self.n2idx[self.final_uid[i]]
        assert self.final_uid[i] == self.captions[0][idx]
        latent = torch.load(os.path.join(self.latent_code_path,self.captions[0][idx]+'.pt')).squeeze()

        return {'caption': self.captions[1][idx], 'latent': latent}

class shapE_val_dataset(Dataset):
    def __init__(self, latent_code_path):
        self.captions = pd.read_csv('./example_material/Cap3D_automated_Objaverse.csv', header=None)
        self.valid_uid = list(pickle.load(open('./example_material/validation_set.pkl','rb')))
        self.final_uid = self.valid_uid
        self.n2idx = {}
        for i in range(len(self.captions)):
            self.n2idx[self.captions[0][i]] = i
        self.latent_code_path = latent_code_path

    def __len__(self):
        return len(self.final_uid)

    def __getitem__(self, i):
        idx = self.n2idx[self.final_uid[i]]
        assert self.final_uid[i] == self.captions[0][idx]
        latent = torch.load(os.path.join(self.latent_code_path,self.captions[0][idx]+'.pt')).squeeze()

        return {'caption': self.captions[1][idx], 'latent': latent}

def train(rank, args):
    if args.gpus > 1:
        setup_ddp(rank, args)

    niter = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    save_name = args.save_name
    f = open('./logs/%s.csv'%save_name, 'a')
    writer = csv.writer(f)

    torch.manual_seed(rank+int(learning_rate*1e6)+int(datetime.now().timestamp()))

    resume_flag = True if args.resume_name != 'none' else False
    if resume_flag:
        model_list = glob.glob('./model_ckpts/%s*.pth'%save_name)
        idx_rank = []
        for l in model_list:
            idx_rank.append(int(l.split('/')[-1].split('_')[-2][5:]) * 21000 + int(l.split('/')[-1].split('_')[-1].split('.')[0]))
        newest = np.argmax(np.array(idx_rank))
        args.resume_name = model_list[newest].split('/')[-1].split('.')[0]

    start_epoch = 0 if not resume_flag else int(args.resume_name.split('_')[-2][5:])
    start_iter = 0 if not resume_flag else int(args.resume_name.split('_')[-1].split('.')[0])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if resume_flag:
        print('reload from ./model_ckpts/%s.pth'%args.resume_name)
        checkpoint = torch.load('./model_ckpts/%s.pth'%args.resume_name, map_location=device)

    if not resume_flag:
        model = load_model('text300M', device=device)
    else:
        model = model_from_config(load_config('text300M'), device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    if args.gpus > 1:
        model = DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=False
        )
    
    diffusion = diffusion_from_config(load_config('diffusion'))
    my_dataset_train = shapE_train_dataset(args.latent_code_path)
    data_loader = DataLoader(my_dataset_train, batch_size=batch_size, num_workers=8, prefetch_factor=4, shuffle=True, drop_last=True)
    my_dataset_val = shapE_val_dataset(args.latent_code_path)
    data_loader_val = DataLoader(my_dataset_val, batch_size=batch_size, num_workers=8, prefetch_factor=4, drop_last=True)


    optimizer= optim.AdamW(model.parameters(), lr=learning_rate)
    total_iter_per_epoch = int(len(my_dataset_train)/batch_size)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, niter*total_iter_per_epoch)
    if resume_flag:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(start_epoch, niter):
        s = time.time()
        for i, data in enumerate(data_loader):
            if i + start_iter == total_iter_per_epoch:
                start_iter = 0
                break
            s2 = time.time()
            prompt = data['caption']
            model_kwargs=dict(texts=prompt)
            t = torch.randint(0, load_config('diffusion')['timesteps'], size=(batch_size,), device=device) 
            x_start = data['latent'].cuda()

            optimizer.zero_grad()
            loss = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)
            final_loss = torch.mean(loss['loss'])

            skip_step = torch.isnan(final_loss.detach()) or not torch.isfinite(final_loss.detach())
            skip_step_tensor = torch.tensor(skip_step, dtype=torch.int).to(device)
            if args.gpus > 1:
                dist.all_reduce(skip_step_tensor, op=dist.ReduceOp.SUM)
            skip_step = skip_step_tensor.item() > 0
            if skip_step:
                del final_loss
                torch.cuda.empty_cache()
            else:
                final_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if args.gpus == 1 or (args.gpus >1 and dist.get_rank() == 0):
                    print('rank: ',rank,time.time()-s2,' epoch: ', epoch, i, final_loss.item())
                if (i+1)%400 == 0:
                    if args.gpus > 1:
                        torch.save({'model_state_dict': model.module.state_dict(), 
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': lr_scheduler.state_dict(),
                                    }, './model_ckpts/%s_epoch%d_%d.pth'%(save_name, epoch, i+start_iter))
                    else:
                        torch.save({'model_state_dict': model.state_dict(), 
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': lr_scheduler.state_dict(),
                                    }, './model_ckpts/%s_epoch%d_%d.pth'%(save_name, epoch, i+start_iter))
                if i%4000 == 0:
                    with torch.no_grad():
                        val_loss = []
                        for j, dataval in enumerate(data_loader_val):
                            prompt = data['caption']
                            model_kwargs=dict(texts=prompt)
                            t = torch.randint(0, load_config('diffusion')['timesteps'], size=(batch_size,), device=device) 
                            x_start = data['latent'].cuda()
                            loss = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)
                            final_loss = torch.mean(loss['loss'])
                            print('validation %d/%d: '%(j, len(data_loader_val)), final_loss.item())
                            val_loss.append(final_loss.item())
                        val_mean_loss = torch.mean(torch.Tensor(val_loss)).item()
                        writer.writerow([epoch, i+start_iter, val_mean_loss])
                        f.flush()
                        os.fsync(f.fileno())
                        print('rank: ',rank, i, val_mean_loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_group = parser.add_argument_group('Model settings')
    model_group.add_argument('--port', type = str, default = '12356', help = 'port for parallel')
    model_group.add_argument('--gpus', type = int, default = 1, help = 'how many gpu use')
    model_group.add_argument('--resume_name', type = str, default = 'none', help = 'any name different from "none" will resume the training')
    model_group.add_argument('--save_name', type = str, default = 'none', help = 'name for the save file')
    model_group.add_argument('--lr', type = float, default = 1e-5, help = 'learning rate')
    model_group.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    model_group.add_argument('--epoch', type = int, default = 25, help = 'total epoch')
    model_group.add_argument('--latent_code_path', type = str, default = './Cap3D_latentcodes', help = 'the directory to the .pt file which store Shap-E latent codes')


    args = parser.parse_args()

    if args.gpus == 1:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))


