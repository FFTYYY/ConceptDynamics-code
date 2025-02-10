from .mlp import Model as Model_MLP
from xingyun import MyDict

models = {
    "MLP": Model_MLP
}

def get_model(model_name: str, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, C: MyDict):
    return models[model_name](input_dim, output_dim, hidden_dim, num_layers, C)
