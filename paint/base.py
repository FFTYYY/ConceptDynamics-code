from xingyun import ArgumentParser
from config import PROJECT
import torch as tc
import torch.nn as nn
from tqdm import tqdm 
from torch.autograd.functional import jacobian

def get_config():
    par = ArgumentParser()
    par.add_argument("exp_id", type = str, default = "9", aliases = ["exp"])
    par.add_argument("info"  , type = str, default = "")
    par.add_argument("device", type = str, default = "cuda:0")
    par.add_argument("folder_prefix", type = str, default = "")

    return par.parse()

def get_input_jaccobian(model:nn.Module, X: tc.Tensor, desc: str = ""):
    '''X: (n,d)
    jacobian: (d_out, d_in)
    '''

    n,d = X.size()
    model.zero_grad()

    # J = jacobian(model, X) # (n,d,n,d)
    # jacob = J[tc.arange(n),:,tc.arange(n),:]
    
    jacobs = []
    for i in tqdm(range(X.size(0)), desc = desc):
        model.zero_grad()
        jacobs.append(jacobian(model , X[i:i+1]).view(1,-1,d))
    jacob = tc.cat(jacobs, dim = 0) # (n,d,d)

    return jacob
