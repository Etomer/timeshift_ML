import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wav
import os
import pandas as pd
import matplotlib.pyplot as plt
import torchaudio
import glob
import math
import torch.nn as nn
import numpy as np
from bisect import bisect_left
import scipy as sp
import h5py
from scipy.io import wavfile
from glob import glob
from contextlib import nullcontext
import shutil
import json
import importlib
import sys

import wandb

use_wandb = True # Note, only set to False when testing files, models will not be saved properly without this


data_folder = "./data/datasets/"
reference_sound_folder = "./data/reference_data/reference_sounds/"
model_folder = "./models/"

device = 'cuda:1' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')



config = {
    "dataset" : "impulse_response_medium.hdf5",

    # Network model config
    "model" : "cnn_model_block",
    "cnn_output_size_at_factor_1" : 576,
    "scale_factor" : 10,
    "continue_training_from_checkpoint" : False, #"./models/new_type_dataset_medium_divine-shadow-51_190.pth",
    "dropout" : 0.1,
    "guess_grid_size" : 500,
    
    # problem specific config
    "max_freq" : 2500, #component
    "rir_len" : 1600,
    "sample_length" : 10000,
    "max_shift" : 100, # ~10 meters
    
    # Optimization config
    "loss_fn" : "cross_entropy",
    "optimizer" : "adamw",
    "lr" : 1e-4,
    "epochs" : 1000,
    "warmup_steps_per_epoch" : 5,
    "warmup_epochs" : 2,
    "rooms_per_batch" : 50,
    "mics_per_batch" : 11,
    "n_batch_before_print" : 3,
}
sys.path.append(model_folder)
model_file = importlib.import_module((config["model"]))

schedule_steps = config["warmup_epochs"]*config["warmup_steps_per_epoch"]

with wandb.init(config=config) if use_wandb else nullcontext() as run:

    # Create run folder (overwrite old one if already exists one with the same name)
    if use_wandb:
        run_folder =  os.path.join(".", "runs" ,run.name)
    else:
        run_folder = os.path.join(".", "runs" , "non_wandb")
    if os.path.isdir(run_folder):
        shutil.rmtree(run_folder)
    os.mkdir(run_folder)
    
    # create checkpoint folder
    os.mkdir(os.path.join(run_folder, "checkpoints"))

    # Save config file
    with open(os.path.join(run_folder,'config.json'), 'w') as f:
        json.dump(config,f)

    # Save copy of model.py in run_folder
    shutil.copyfile(os.path.join(model_folder,config["model"]+".py"), os.path.join(run_folder,config["model"]+".py"))
    
    with h5py.File(data_folder + config["dataset"],"r") as f:

        X = f['input']
        y = f['gt']

        class custom_dataset(Dataset):

            def __init__(self, X, y, idx_min=0,dataset_len=len(X)):
                self.X = X
                self.y = y
                self.dataset_len = dataset_len
                self.idx_min = idx_min

            def __getitem__(self, idx):

                mics_idx = torch.randperm(X.shape[1])[:config["mics_per_batch"]].sort()[0] # sorting because hdf5 requires it
                return self.X[idx + self.idx_min, mics_idx],self.y[idx + self.idx_min, mics_idx]

            def __len__(self):
                return self.dataset_len

        # package datasets
        split_i = int(X.shape[0]*0.9)
        dataset = custom_dataset(X,y, 0, split_i)
        dataset_test = custom_dataset(X,y, split_i, X.shape[0] - split_i)
        train_dl = DataLoader(dataset, batch_size=config["rooms_per_batch"], shuffle=True)
        test_dl = DataLoader(dataset_test, batch_size=config["rooms_per_batch"], shuffle=False)


            
        model = model_file.model(config).to(device)

        if config["continue_training_from_checkpoint"]:
            model = torch.load(config["continue_training_from_checkpoint"])

        if config["loss_fn"] == "cross_entropy":
            loss_fn = torch.nn.CrossEntropyLoss()
        

        def y_to_class_gt(y, guess_grid_size, max_shift) :
            y[y.abs() > max_shift] = max_shift*y[y.abs() > max_shift].sign()

            bin_width = max_shift*2/guess_grid_size
            y = (y/bin_width).round() + guess_grid_size // 2 
            y[y == guess_grid_size] = guess_grid_size - 1
            return y.long()


        def format_simulated_data(X,y):
            """
            transform a tensor of impulse responses in different rooms into pairwise TimeEstimation-problems. Note (X and y should be on GPU)

            """
            #pull a random sound
            sound_paths = glob(os.path.join(reference_sound_folder, "*.wav"))
            sound_path = sound_paths[np.random.randint(len(sound_paths))]
            fs,signal = wavfile.read(sound_path)
            start = np.random.randint(0,len(signal) - config["sample_length"] - config["rir_len"]-1, X.shape[0])
            # simulate longer sound and then cut to the relevant piece
            signals = np.zeros((X.shape[0], config["sample_length"] + config["rir_len"]-1))
            for i in range(X.shape[0]):
                signals[i,:] = signal[start[i]:start[i]  + config["sample_length"] + config["rir_len"]-1]
            #signals = torch.tensor(signals).to(torch.float32).to(device).unsqueeze(1)
            signals = torch.tensor(signals).to(torch.float32).unsqueeze(1)

            q = torch.fft.irfft(torch.fft.rfft(signals)*torch.fft.rfft(X,signals.shape[2]))[:,:,:config["sample_length"]] # compute the heard sound, and cut it to the right length
            q = torch.fft.rfft(q)[:,:,:config["max_freq"]] # cut frequencies which are too high
            q = q.unsqueeze(2)
            q = torch.concatenate([torch.concatenate([q,q.roll(i+1, 1)], dim=2) for i in range(config["mics_per_batch"] // 2)],dim=1) # organize sounds pairwise
            q = q.view(X.shape[0]*(config["mics_per_batch"]*(config["mics_per_batch"] - 1 ))//2, 2,-1) # reshape so that each example is a row
            X = torch.concatenate([q.real,q.imag],dim=1)
            X /= X.std(dim=2).mean(dim=1).unsqueeze(1).unsqueeze(2) + 1e-5 # avoid dividing by 0
            y = torch.concatenate([y - y.roll(i+1,1) for i in range(config["mics_per_batch"]//2)],dim=1).view(-1)*fs/343 # compute gt for all pairs
            y = y_to_class_gt(y, config["guess_grid_size"], config["max_shift"]).to(torch.long)
            
            return X,y


        def train(dataloader, model, loss_fn, optimizer, scheduler, warm_up=False):
            size = len(dataloader.dataset)
            model.train()
            print_loss = 0
            
            
            for batch, (X,y) in enumerate(dataloader):
                # Compute prediction error
                batch_rooms = X.shape[0]
                X,y = format_simulated_data(X,y)
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #increase learning rate if in wamup
                if warm_up and batch%(1 + (len(dataloader)//config["warmup_steps_per_epoch"])) == 0:
                    scheduler.step()
                    
                    

                #printing and logging
                print_loss += loss.detach()/config["n_batch_before_print"]
                if batch % config["n_batch_before_print"] == 0:
                    if batch == 0:
                        print_loss = 0 
                        continue
                    if use_wandb:
                        wandb.log({"loss":loss})
                    loss, current = print_loss.item(), batch * batch_rooms
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    print_loss = 0
            

            
        def test(dataloader, model, loss_fn):
            model.eval() # regression on dropout is not great
            
            with torch.no_grad():
                print_loss = 0
                counter = 0
                for batch, (X,y) in enumerate(dataloader):
                    
                    # Compute prediction error
                    X,y = format_simulated_data(X,y)
                    
                    
                    X = X.to(device)
                    y = y.to(device)
                    pred = model(X)
                    loss = loss_fn(pred, y)
                        
                    print_loss += loss.detach()
                    counter += 1


                loss = print_loss.item()/counter
                if use_wandb:
                    wandb.log({"Test_loss":loss})
                print(f"Test loss: {loss:>7f}")
                print_loss = 0




        if config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"]/2**schedule_steps)
        elif config["optimizer"] == "radam":
            optimizer = torch.optim.RAdam(model.parameters(), lr=config["lr"]/2**schedule_steps)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)
        if use_wandb:
            wandb.watch(model, log_freq=50)

        for t in range(config["epochs"]):
            losses = np.zeros((0,2))
            print(f"Epoch {t+1}\n-------------------------------")

            if t == config["warmup_epochs"]:
                print(f'WarmUp done! now using lr = {optimizer.param_groups[0]["lr"]:.2}')
                

            test(test_dl, model, loss_fn)
            
            train(train_dl, model, loss_fn, optimizer,scheduler, warm_up=(t<config["warmup_epochs"]))
            
            if t % 100 == 0:
                if use_wandb:
                    #torch.save(model, "./models/" + config["dataset"].split(".")[0]+ "_" + run.name + "_" + str(t) +".pth")
                    torch.save(model, os.path.join(run_folder, "checkpoints" ,run.name + "_" + str(t) +".pth"))
                else:
                    #torch.save(model, "./models/" + config["dataset"].split(".")[0]+ "_not_logged_" + str(t) +".pth")
                    torch.save(model, os.path.join(run_folder,"checkpoints", "not_logged_" + str(t) +".pth"))

        print("Done!")
