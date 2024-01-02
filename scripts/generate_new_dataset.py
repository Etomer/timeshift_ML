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


sample_length = 10000
extra_bit = 1000

reflection_coeff = 0.05
scatter_coeff = 0.15

speed_of_sound = 343


fs = 16000
max_freq = 4000
max_freq_component_to_use = int(max_freq / (fs / sample_length))


n_rooms = 200
n_mics = 51
rir_len = 1600

fs = 16000

# X = np.zeros((n_rooms,n_mics,rir_len))
# Y = np.zeros((n_rooms,n_mics))

with h5py.File(
    "./datasets/generated_dataset/new_type_dataset_realistic_evaluation.hdf5", "w"
) as hdf5_file:
    X = hdf5_file.create_dataset("input", (n_rooms, n_mics, rir_len), dtype="f")
    Y = hdf5_file.create_dataset("gt", (n_rooms, n_mics), dtype="f")

    for room_i in range(n_rooms):
        # randomly generate a rectangular cuboid
        x, y, z = 9 * np.random.rand(3) + 1
        corners = np.array([[0, 0], [0, y], [x, y], [x, 0]]).T
        room = pra.Room.from_corners(
            corners,
            fs=fs,
            max_order=2,
            materials=pra.Material(reflection_coeff, scatter_coeff),
            ray_tracing=True,
            air_absorption=True,
        )
        room.extrude(z, materials=pra.Material(reflection_coeff, scatter_coeff))
        room.set_ray_tracing(receiver_radius=0.2, n_rays=10000, energy_thres=1e-5)

        # add sender and receivers to room
        random_point_in_room = lambda: np.random.rand(3) * [x, y, z]
        sender_position = random_point_in_room()
        room.add_source(sender_position)
        R = np.array(np.stack([random_point_in_room() for i in range(n_mics)]).T)
        room.add_microphone(R)

        # compute image sources for reflections
        room.image_source_model()
        room.compute_rir()

        for mic_i in range(n_mics):
            if len(room.rir[mic_i][0]) > rir_len:
                X[room_i, mic_i] = room.rir[mic_i][0][:rir_len]
                Y[room_i, mic_i] = np.linalg.norm(sender_position - R[:, mic_i])
            else:
                X[room_i, mic_i, : len(room.rir[mic_i][0])] = room.rir[mic_i][0]
                Y[room_i, mic_i] = np.linalg.norm(sender_position - R[:, mic_i])
