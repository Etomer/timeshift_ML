import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
from glob import glob
import scipy as sp
import torch
import h5py


batch_simulate_at_same_locations = 25
n_batches = 200
dataset_size = n_batches*batch_simulate_at_same_locations

# constants
sample_length = 10000
extra_bit = 1000

reflection_coeff = 0.05
scatter_coeff = 0.15

speed_of_sound = 343


fs = 16000
max_freq = 4000
max_freq_component_to_use = int(max_freq/(fs/sample_length))


with h5py.File("./datasets/generated_dataset/generated_dataset_val.hdf5","w") as hdf5_file:

    X = hdf5_file.create_dataset("input", (dataset_size,2,max_freq_component_to_use,2), dtype="f")
    Y = hdf5_file.create_dataset("gt", (dataset_size,1), dtype="f")

    #X = torch.zeros(dataset_size,2,max_freq_component_to_use,2)
    #Y = torch.zeros(dataset_size,1)
    for ii in range(n_batches):



        # load audio
        sound_paths = glob("./datasets/reference_sounds/*.wav")
        sound_path = sound_paths[np.random.randint(len(sound_paths))]

        fs,signal = wavfile.read(sound_path)
        start = np.random.randint(len(signal) - sample_length*batch_simulate_at_same_locations - extra_bit)
        # simulate longer sound and then cut to the relevant piece
        signal = signal[start:start + batch_simulate_at_same_locations*sample_length + extra_bit]



        # randomly generate a rectangular cuboid
        x,y,z = 9*np.random.rand(3) + 1
        corners = np.array([[0,0], [0,y], [x,y], [x,0]]).T 
        room = pra.Room.from_corners(corners, fs=fs, max_order=2, materials=pra.Material(reflection_coeff, scatter_coeff), ray_tracing=True, air_absorption=True)
        room.extrude(z, materials=pra.Material(reflection_coeff, scatter_coeff))
        room.set_ray_tracing(receiver_radius=0.2, n_rays=10000, energy_thres=1e-5)

        #add sender and receivers to room
        random_point_in_room = lambda : np.random.rand(3)*[x,y,z]
        sender_position = random_point_in_room()
        receiver1_position = random_point_in_room()
        receiver2_position = random_point_in_room()
        room.add_source(sender_position, signal=signal)
        R = np.array(np.stack([receiver1_position,receiver2_position]).T)
        room.add_microphone(R)

        # Ground truth value
        gt_tdoa = (np.linalg.norm(sender_position - receiver1_position) - np.linalg.norm(sender_position - receiver2_position))*fs/speed_of_sound

        # compute image sources for reflections
        room.image_source_model()

        #simulate room
        room.simulate()
        
        
        for i in range(batch_simulate_at_same_locations):
            new_example_index = ii*batch_simulate_at_same_locations + i
            temp = sp.fft.fft((room.mic_array.signals[0,extra_bit + (i)*sample_length:extra_bit + (i+1)*sample_length]))
            X[new_example_index,0,:,0] = torch.tensor(np.real(temp[:max_freq_component_to_use]))
            X[new_example_index,1,:,0] = torch.tensor(np.imag(temp[:max_freq_component_to_use]))
            temp = sp.fft.fft((room.mic_array.signals[1,extra_bit + (i)*sample_length:extra_bit + (i+1)*sample_length]))
            X[new_example_index,0,:,1] = torch.tensor(np.real(temp[:max_freq_component_to_use]))
            X[new_example_index,1,:,1] = torch.tensor(np.imag(temp[:max_freq_component_to_use]))
            #X[new_example_index,1] = torch.tensor(room.mic_array.signals[1,extra_bit + (i)*sample_length:extra_bit + (i+1)*sample_length])
            Y[new_example_index, 0:1] = torch.tensor(gt_tdoa)


        #IPython.display.Audio(room.mic_array.signals[0,:], rate=fs)

    #torch.save(X,"./datasets/generated_dataset/input.pt")
    #torch.save(Y,"./datasets/generated_dataset/gt.pt")