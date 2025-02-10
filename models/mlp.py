import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from xingyun import MyDict
import copy 

class Model(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, C: MyDict):
        super(Model, self).__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.relu = C["relu"]
        self.no_bias = C["no_bias"]

        self.layers = nn.ModuleList([])
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim, bias = not self.no_bias))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias = not self.no_bias))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias = not self.no_bias))
            self.layers.append(nn.Linear(hidden_dim, output_dim, bias = not self.no_bias))
    
    def reset_params(self):
        for layer in self.layers:
            nn.init.normal_( layer.weight.data  , 0 , 1e-4)
            nn.init.constant_( layer.bias.data , 0 )

    def forward(self, X):
        '''X: (n, input_dim)'''
        for layer_idx, layer in enumerate(self.layers):
            X = layer(X)
            if self.relu and layer_idx < len(self.layers) - 1:
                X = F.relu(X)
        return X
    
    # @tc.no_grad()
    # def signs(self, x: tc.Tensor):

    #     if not self.relu:
    #         raise ValueError("fuck you") 

    #     z = x
    #     signs = []
    #     for layer_idx , layer in enumerate( self.layers ):
    #         z = layer(z)
    #         if layer_idx < len(self.layers) - 1:
    #             D = (z >= 0).float()
    #             signs.append(D.detach().clone())
    #             z = z * D

    #     assert tc.norm( z - self.forward(x) ) <= 1e-5

    #     return tc.cat(signs, dim = -1)

    @tc.no_grad()
    def signs(self, x: tc.Tensor):

        z = x
        signs = []
        for layer_idx , layer in enumerate( self.layers ):
            z: tc.Tensor = layer(z)
            if layer_idx < len(self.layers) - 1:
                if self.relu:
                    D = (z >= 0).float()
                else:
                    D = z.new_ones(z.size())
                signs.append(D.detach().clone())
                z = z * D

        assert tc.norm( z - self.forward(x) ) <= 1e-5

        return tc.cat(signs, dim = -1)
    
    @tc.no_grad()
    def perturb(self, var: float):
        another_me = copy.deepcopy(self)

        for w in another_me.parameters():
            w.data = w.data + tc.randn(w.data.size()).to(w.device) * var
        
        return another_me
        

        

