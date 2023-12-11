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

import wandb

use_wandb = True


data_folder = "./datasets/generated_dataset/"
device = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')



config = {
    "dropout" : 0.1,
    "batch_size" : 11, 
    "max_shift" : 500, # ~10 meters
    "guess_grid_size" : 100,
    "dataset" : "new_type_dataset_medium.hdf5",
    "cnn_output_size_at_factor_1" : 576,
    "factor" : 10,
    "loss_fn" : "cross_entropy",
    "epochs" : 200,
    "sample_length" : 10000,
    "max_shift" : 100,
    "lr" : 3e-4,
    "n_batch_before_print" : 10,
    "max_freq" : 2500,
    "rir_len" : 1600,
    "rooms_per_batch" : 50,
    "mics_per_batch" : 11,
    "warmup_steps_per_epoch" : 5,
    "warmup_epochs" : 2,
}

schedule_steps = config["warmup_epochs"]*config["warmup_steps_per_epoch"]

wandb.init()

f = h5py.File(data_folder + config["dataset"],"r")

X = f['input']
y = f['gt']

class custom_dataset(Dataset):

    def __init__(self, X, y, idx_min=0,dataset_len=len(X)):
        self.X = X
        self.y = y
        self.dataset_len = dataset_len
        self.idx_min = idx_min

    def __getitem__(self, idx):
        return self.X[idx + self.idx_min,:config["mics_per_batch"]],self.y[idx + self.idx_min,:config["mics_per_batch"]]

    def __len__(self):
        return self.dataset_len

# package datasets
split_i = int(X.shape[0]*0.9)
dataset = custom_dataset(X,y, 0, split_i)
dataset_test = custom_dataset(X,y, split_i, X.shape[0] - split_i)
train_dl = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
test_dl = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=False)


class Block(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.dropout = nn.Dropout(config["dropout"])
        self.l = nn.Linear(size,2*size)
        self.l2 = nn.Linear(2*size,size)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.002)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.002)
            
    def forward(self, x):
        return x + self.l2(self.act(self.l(self.ln(self.dropout(x)))))

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.thinker = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["factor"]*config["cnn_output_size_at_factor_1"],1000),
            nn.GELU(),
            Block(1000),
            Block(1000),
            nn.Linear(1000,config["guess_grid_size"])
        )
        
        self.apply(self._init_weights)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(4,48*config["factor"], 50,stride=5),
            nn.GELU(),
            nn.Conv1d(48*config["factor"],48*config["factor"], 50,stride=5),
            nn.GELU(),
            nn.Conv1d(48*config["factor"],48*config["factor"], 30,stride=5),
            nn.GELU(),
            nn.Flatten(),
        )
        
        
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.0002)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.0002)
        
                
    def forward(self, x):
        x = self.cnn(x)
        x = self.thinker(x)
        return x
    
model = Classifier().to(device)

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
    sound_paths = glob("./datasets/reference_sounds/*.wav")
    sound_path = sound_paths[np.random.randint(len(sound_paths))]
    fs,signal = wavfile.read(sound_path)
    start = np.random.randint(0,len(signal) - config["sample_length"] - config["rir_len"]-1, X.shape[0])
    # simulate longer sound and then cut to the relevant piece
    signals = np.zeros((X.shape[0], config["sample_length"] + config["rir_len"]-1))
    for i in range(X.shape[0]):
        signals[i,:] = signal[start[i]:start[i]  + config["sample_length"] + config["rir_len"]-1]
    signals = torch.tensor(signals).to(torch.float32).to(device).unsqueeze(1)

    q = torch.fft.irfft(torch.fft.rfft(signals)*torch.fft.rfft(X,signals.shape[1]))[:,:,:config["sample_length"]] # compute the heard sound, and cut it to the right length
    q = torch.fft.rfft(q)[:,:,:config["max_freq"]] # cut frequencies which are too high
    q = q.unsqueeze(2)
    q = torch.concatenate([torch.concatenate([q,q.roll(i+1, 1)], dim=2) for i in range(config["mics_per_batch"] // 2)],dim=1) # organize sounds pairwise
    q = q.view(X.shape[0]*(config["mics_per_batch"]*(config["mics_per_batch"] - 1 ))//2, 2,-1) # reshape so that each example is a row
    X = torch.concatenate([q.real,q.imag],dim=1)
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
        X = X.to(device)
        
        y = y.to(device)
        X,y = format_simulated_data(X,y)
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
            X = X.to(device)
            y = y.to(device)
            X,y = format_simulated_data(X,y)
            pred = model(X)
            loss = loss_fn(pred, y)
                
            print_loss += loss.detach()
            counter += 1


        loss = print_loss.item()/counter
        if use_wandb:
            wandb.log({"Test_loss":loss})
        print(f"Test loss: {loss:>7f}")
        print_loss = 0





optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"]/2**schedule_steps)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)

wandb.watch(model, log_freq=50)

for t in range(config["epochs"]):
    losses = np.zeros((0,2))
    print(f"Epoch {t+1}\n-------------------------------")

    test(test_dl, model, loss_fn)
    
    train(train_dl, model, loss_fn, optimizer,scheduler, warm_up=(t<config["warmup_epochs"]))
    
    if t % 10 == 0:
        torch.save(model, "./models/" + config["dataset"].split(".")[0]+ "_" + run.name + "_" + str(t) +".pth")

print("Done!")
