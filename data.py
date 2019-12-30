"""
All data-related functions.
"""
import numpy as np
import matplotlib.pyplot as plt

import torch

import contextlib
import itertools
import os


##################
# Generating data.
##################

def generate_sinusoidal(A=1.0, w=1.0, phi=0.0, c=0.0, x_shift=0.0, N=25, plot=True):
    # Generate a curve of form y = A*sin(w*x + phi) + c for N x values linearly spaced in [0 + shift, 2*pi + shift]
    
    x = np.linspace(0 + x_shift, 2*np.pi + x_shift, N)
    y = A*np.sin(w*x + phi) + c

    if plot:
        # Plot a smooth line for visualisation.
        x_ = np.linspace(0 + x_shift, 2*np.pi + x_shift, 500)
        y_ = A*np.sin(w*x_ + phi) + c
        plt.plot(x_, y_)
        
        # Plot data.
        plt.scatter(x, y)
    
    return x, y


def generate_data(A_s, w_s, phi_s, c_s, x_shifts, plot=True, title="Generated Curves"):
    # Generate the curves based on all the combinations of A_s, w_s, phi_s, c_s, x_shifts.

    prod = itertools.product(A_s, w_s, phi_s, c_s)

    plt.figure(figsize=(25, 10), dpi=70)
    arrdict = dict()

    for idx, (A, w, phi, c) in enumerate(prod):
        x, y = generate_sinusoidal(A=A, w=w, phi=phi, c=c, x_shift=x_shifts[idx], plot=True)
        xname = "x_{}".format(idx)
        yname = "y_{}".format(idx)
        arrdict[xname] = x
        arrdict[yname] = y
    
    if plot:
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
    
    return arrdict


def generate_and_save_datasets(data_root="./data"):

    # Make directory.
    data_root_realpath = os.path.realpath(data_root)
    if not os.path.exists(data_root_realpath):
        os.makedirs(data_root_realpath)

    # Meta-train dataset.

    A_s = (0.2, 1.0, 2.5, 5.0)
    w_s = (0.25*np.pi, 1.0, 3.0*np.pi)
    phi_s = (-0.5*np.pi, 1.0, 2.0*np.pi)
    c_s = (-2.0, 0.0, 4.0)

    np.random.seed(12345)
    n_combs = len(A_s) * len(w_s) * len(phi_s) * len(c_s)
    x_shifts = np.random.randn(n_combs) * 2*np.pi

    arrdict = generate_data(A_s, w_s, phi_s, c_s, x_shifts, plot=True, title="Generated Curves: Meta-train")

    data_path_meta_train = os.path.realpath(os.path.join(data_root, "data_curves_meta-train.npz"))
    np.savez(data_path_meta_train, **arrdict)

    # Meta-test dataset.

    A_s = (0.8, 4.3)
    w_s = (0.5*np.pi, 1.3*np.pi, 2.3*np.pi)
    phi_s = (-0.3*np.pi, 1.2*np.pi, 4.3*np.pi)
    c_s = (-1.5, 2.0)

    np.random.seed(54321)
    n_combs = len(A_s) * len(w_s) * len(phi_s) * len(c_s)
    x_shifts = np.random.randn(n_combs) * 2*np.pi

    arrdict = generate_data(A_s, w_s, phi_s, c_s, x_shifts, plot=True, title="Generated Curves: Meta-test")

    data_path_meta_test = os.path.realpath(os.path.join(data_root, "data_curves_meta-test.npz"))
    np.savez(data_path_meta_test, **arrdict)
    
    return data_path_meta_train, data_path_meta_test


def load_data(data_path):
    data = np.load(data_path)
    names = data.files
    l = len(names) // 2
    x, y = dict(), dict()
    for idx in range(l):
        x_key = "x_{}".format(idx)
        y_key = "y_{}".format(idx)
        x[idx] = data[x_key]
        y[idx] = data[y_key]
    return x, y



#################
# Providing data.
#################

@contextlib.contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class TrainTestSplitter(object):
    # Train / test splitter for each item inside the meta dataset.
    
    def __init__(self, test_frac=0.4, seed=12345):
        self.seed = seed
        self.test_frac = test_frac
        self.train_frac = 1 - test_frac
    
    def __call__(self, dataset):
        length = len(dataset)
        train_end_idx = int(np.floor(length * self.train_frac))
        indices = np.array(range(length))
        with temp_np_seed(self.seed):
            np.random.shuffle(indices)
        train_indices = indices[:train_end_idx]
        test_indices = indices[train_end_idx:]
        return np.take(dataset, train_indices), np.take(dataset, test_indices)


class CurveTasks(torch.utils.data.Dataset):
    # Meta Dataset.
    
    def __init__(self, train_test_splitter, data_root="./data", meta_train=True):
        if meta_train:
            data_path = os.path.realpath(os.path.join(data_root, "data_curves_meta-train.npz"))
        else:
            data_path = os.path.realpath(os.path.join(data_root, "data_curves_meta-test.npz"))
        self.x, self.y = load_data(data_path)
        self.train_test_splitter = train_test_splitter
    
    def __getitem__(self, key):
        x_train, x_test = self.train_test_splitter(self.x[key])
        y_train, y_test = self.train_test_splitter(self.y[key])
        return (x_train, y_train), (x_test, y_test)
    
    def __len__(self):
        return len(self.x)
