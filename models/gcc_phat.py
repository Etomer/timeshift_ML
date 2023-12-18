#includes gcc-phat packaged in the format of a pytorch model

import torch


    
def model(config):
    def forward(x):
        x = torch.complex(x[:,0::2,:],x[:,1::2])
        c = x[:,0]*x[:,1].conj()
        x = torch.fft.irfft(torch.concatenate([c/(c.abs() + 1e-10), torch.zeros(x.shape[0], 1+ config["sample_length"]//2 - x.shape[2])], dim=1))
        
        x = torch.fft.fftshift(x,dim=1)
        # recompute  correct bin size
        bin_size = 2*config["max_shift"]/config["guess_grid_size"]
        pred = torch.zeros(x.shape[0], config["guess_grid_size"])
        for i in range(config["guess_grid_size"]):
            start = int(x.shape[1] // 2 - (config["guess_grid_size"] / 2)*bin_size + bin_size * i)
            #print(torch.sum(x[:,start:(start + bin_size)],dim=1).shape)
            pred[:,i] = torch.sum(x[:,start:(start + int(bin_size))],dim=1)

        return pred
    return forward


## old version
#import torch.nn as nn
# class gcc_phat_model(nn.Module):

#     def __init__(self, sample_length, max_shift, grid_guess_size):
#         self.sample_length = sample_length
#         self.max_shift = max_shift
#         self.grid_guess_size = grid_guess_size

#     def forward(self, x):
#         x = torch.complex(x[:,0::2,:],x[:,1::2])
#         c = x[:,0]*x[:,1].conj()
#         x = torch.fft.irfft(torch.concatenate([c/(c.abs() + 1e-10), torch.zeros(x.shape[0], 1+ self.sample_length//2 - x.shape[2])], dim=1))
        
#         x = torch.fft.fftshift(x,dim=1)
#         # recompute  correct bin size
#         bin_size = 2*self.max_shift/self.grid_guess_size
#         pred = torch.zeros(x.shape[0], self.grid_guess_size)
#         for i in range(self.grid_guess_size):
#             start = int(x.shape[1] // 2 - (self.grid_guess_size / 2)*bin_size + bin_size * i)
#             #print(torch.sum(x[:,start:(start + bin_size)],dim=1).shape)
#             pred[:,i] = torch.sum(x[:,start:(start + int(bin_size))],dim=1)

#         return pred


    

