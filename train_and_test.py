import torch as tc
import torch.nn as nn
import torch.nn.functional as F 
from xingyun import MyDict
from tqdm import tqdm 

optimizers =  {
    "SGD": tc.optim.SGD,
}

def get_optimizer(optimizer: str, C: MyDict, model:nn.Module):
    return optimizers[optimizer](model.parameters(), lr = C["lr"], weight_decay = C["wd"])

def train(model:nn.Module, optimizer: tc.optim.Optimizer, train_X: tc.Tensor, train_Y: tc.Tensor):
    model = model.train()

    optimizer.zero_grad()
    pred_Y = model(train_X)
    loss = F.mse_loss(pred_Y, train_Y)

    loss.backward()
    optimizer.step()
    
    return float(loss), model

@tc.no_grad()
def test(model:nn.Module, test_X: tc.Tensor, test_Y: tc.Tensor, reduce: bool = True):
    model = model.eval()

    pred_Y = model(test_X)
    loss = F.mse_loss(pred_Y, test_Y,reduce = reduce)
    
    return float(loss) if reduce else loss

    

    