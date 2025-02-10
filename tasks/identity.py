'''2d identity mapping learning task.'''


import torch as tc
import numpy as np
from xingyun import GlobalDataManager, MyDict

def map_2d(X: tc.Tensor, Y: tc.Tensor, info: dict, device: str = "cuda:0"):
    U = info["U"].to(device)
    if X is not None:
        X = X.to(device) @ U.t()
    if Y is not None:
        Y = Y.to(device) @ U.t()
    return X, Y

def make_data(C: MyDict): 
    dim = C["dim"]
    dist_1 , dist_2 = C["dists"]

    n_lt = C["n_lt"]
    n_lb = C["n_lb"]
    n_rt = C["n_rt"]
    n_rb = C["n_rb"]
    no_rotation = C["no_rotation"]

    mu_lb = tc.FloatTensor([0       ,   0])
    mu_lt = tc.FloatTensor([0       ,   dist_2])
    mu_rt = tc.FloatTensor([dist_1  ,   dist_2])
    mu_rb = tc.FloatTensor([dist_1  ,   0])

    X_lb = np.random.multivariate_normal(mu_lb, cov = tc.diag(tc.FloatTensor(C["sigma_lb"])).numpy(), size = n_lb) # (n,2)
    X_lt = np.random.multivariate_normal(mu_lt, cov = tc.diag(tc.FloatTensor(C["sigma_lt"])).numpy(), size = n_lt)
    X_rt = np.random.multivariate_normal(mu_rt, cov = tc.diag(tc.FloatTensor(C["sigma_rt"])).numpy(), size = n_rt)
    X_rb = np.random.multivariate_normal(mu_rb, cov = tc.diag(tc.FloatTensor(C["sigma_rb"])).numpy(), size = n_rb)

    X_lb = tc.FloatTensor(X_lb)
    X_lt = tc.FloatTensor(X_lt)
    X_rt = tc.FloatTensor(X_rt)
    X_rb = tc.FloatTensor(X_rb)

    Y_lb = X_lb.clone().detach()  
    Y_lt = X_lt.clone().detach()  
    Y_rt = X_rt.clone().detach()  
    Y_rb = X_rb.clone().detach()  

    train_X = tc.cat([X_lb,X_lt,X_rb], dim = 0) # (3n, 2)
    train_Y = tc.cat([Y_lb,Y_lt,Y_rb], dim = 0)
    test_X = X_rt 
    test_Y = Y_rt

    # --- random shuffle ---
    shuffle_train = tc.randperm(train_X.size(0))
    shuffle_test  = tc.randperm(test_X.size(0))  
    train_X = train_X[shuffle_train]
    train_Y = train_Y[shuffle_train]
    test_X  = test_X[shuffle_test]
    test_Y  = test_Y[shuffle_test]

    # --- make higher dim ---
    U = tc.eye(dim, dim)
    if not no_rotation:
        U = tc.randn(dim , dim)
        U , _ , _ = tc.linalg.svd(U)

    U = U[:2] # (2,dim)
    train_X = train_X @ U
    train_Y = train_Y @ U
    test_X  = test_X @ U
    test_Y  = test_Y @ U

    return train_X, train_Y, test_X, test_Y, {"U": U}


