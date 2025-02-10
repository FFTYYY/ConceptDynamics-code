'''multi-dim identity mapping learning.
'''

import torch as tc
import numpy as np
from xingyun import MyDict

def map_lowdim(X: tc.Tensor, Y: tc.Tensor, num_class:int , info: dict, device: str = "cuda:0"):
    U = info["U"].to(device)
    Uupdim = U[:num_class] # (num_class,dim)

    if X is not None:
        X = X.to(device) @ Uupdim.t()
    if Y is not None:
        Y = Y.to(device) @ Uupdim.t()
    return X, Y

def make_data_cluster(mu: tc.Tensor, sigma: float, num: int):
    s = mu.view(-1).size(0)
    Xs = np.random.multivariate_normal(mu, cov = tc.eye(s) * sigma, size = num) 
    Xs = tc.FloatTensor(Xs)
    Ys = Xs.clone().detach()
    return Xs, Ys

def make_data(C: MyDict): 

    dim = C["dim"]
    num_class = C["num_class"]
    num_sample = C["num_sample"]
    sigma = C["sigma"]
    noise = C["noise"]
    dists = C["dists_multi"]
    no_rotation = C["no_rotation"]

    mus = tc.eye(num_class) * tc.FloatTensor(dists).view(-1)
    train_X, train_Y = [], []
    for _ in range(num_class):
        Xs, Ys = make_data_cluster(mus[_], sigma, num_sample)
        train_X.append(Xs)
        train_Y.append(Ys)
    train_X  = tc.cat(train_X, dim = 0)
    train_Y  = tc.cat(train_Y, dim = 0)

    test_X, test_Y = make_data_cluster(mus.sum(0), sigma, num_sample)

    # --- random shuffle ---
    shuffle_train = tc.randperm(train_X.size(0))
    shuffle_test  = tc.randperm(test_X.size(0))  
    train_X = train_X[shuffle_train]
    train_Y = train_Y[shuffle_train]
    test_X  = test_X [shuffle_test ]
    test_Y  = test_Y [shuffle_test ]

    # --- make higher dim ---
    U = tc.eye(dim, dim)
    if not no_rotation:
        U = tc.randn(dim , dim)
        U , _ , _ = tc.linalg.svd(U)

    Uupdim = U[:num_class] # (num_class,dim)
    train_X = train_X @ Uupdim
    train_Y = train_Y @ Uupdim
    test_X  = test_X @ Uupdim
    test_Y  = test_Y @ Uupdim

    noise_train = tc.randn(train_X.size(0), dim) * noise  # (n,dim)
    noise_train [:,:num_class] = 0 # no noise on first dimensions
    train_X = train_X + noise_train @ U

    noise_test  = tc.randn(test_X.size(0), dim ) * noise  # (n,dim)
    noise_test  [:,:num_class] = 0 # no noise on first dimensions
    test_X = test_X + noise_test @ U


    return train_X, train_Y, test_X, test_Y, {"U": U}


