from xingyun import ArgumentParser, PreCondition, DataAccess
from tasks import tasks
from models import models
from train_and_test import optimizers
import os
import pickle

PROJECT     = "ConceptDynamics"
TASKS       = list(tasks)
MODELS      = list(models)
OPTIMIZERS  = list(optimizers)

def verify_tuple(size : int):
    def _v(s: list | tuple):
        return len(s) == size
    return _v

def type_tuple(s: str):
    return [float(_) for _ in s.split(",")]

def get_config():
    par = ArgumentParser()
    
    par.add_argument("group"   , str , default = "default")
    par.add_argument("seed"    , int , default = 2333)
    par.add_argument("info"    , str , default = "")
    par.add_argument("device"  , str , default = "cuda:0")
    par.add_argument("id"      , int , default = 0)

    par.add_argument("data"    , str , default = "identity", verify = lambda x: x in TASKS)
    par.add_argument("data/dim", int , default = 64 , aliases = ["dim"])
    par.add_bool    ("data/no_rotation", aliases = ["no_rotation"])
    with PreCondition(["data"], lambda d: d in ["identity", "identity_truncate"]):
        par.add_argument("data/n_lt"  , int , default = 5000, aliases = ["n_lt"]) # num. of points in the left-top corner
        par.add_argument("data/n_lb"  , int , default = 5000, aliases = ["n_lb"]) # num. of points in the left-bottom corner
        par.add_argument("data/n_rt"  , int , default = 5000, aliases = ["n_rt"]) # num. of points in the right-top corner
        par.add_argument("data/n_rb"  , int , default = 5000, aliases = ["n_rb"]) # num. of points in the right-bottom corner
        par.add_argument("data/sigma_lt", type_tuple, default = [.05,.05], verify = verify_tuple(2), aliases=["sigma_lt"] ) # sigma of the left-top corner
        par.add_argument("data/sigma_lb", type_tuple, default = [.05,.05], verify = verify_tuple(2), aliases=["sigma_lb"] ) # sigma of the left-bottom corner
        par.add_argument("data/sigma_rt", type_tuple, default = [.05,.05], verify = verify_tuple(2), aliases=["sigma_rt"] ) # sigma of the right-top corner
        par.add_argument("data/sigma_rb", type_tuple, default = [.05,.05], verify = verify_tuple(2), aliases=["sigma_rb"] ) # sigma of the right-bottom corner
        par.add_argument("data/dists"   , type_tuple, default = [2  ,2  ], verify = verify_tuple(2), aliases=["dists"] ) # (mu_x, mu_y)

    with PreCondition(["data"], lambda d: d == "multi_identity"):
        par.add_argument("data/num_class"  , int   , default = 4    , aliases = ["num_class"] ) # num. of classes (dimensions)
        par.add_argument("data/num_sample" , int   , default = 5000 , aliases = ["num_sample"]) # num. of points in each class
        par.add_argument("data/sigma"      , float , default = 0.2  , aliases = ["sigma"]     ) # sigma of the gaussian distribution
        par.add_argument("data/noise"      , float , default = 0    , aliases = ["noise"]     ) # noise of the gaussian distribution
        par.add_argument("data/dists_multi", type_tuple , default = [1,2,3,4]   , aliases = ["dists_multi"]) # mu of each direction
    

    par.add_argument("model"            , str , default = "MLP" , verify = lambda x: x in MODELS)
    par.add_argument("model/hdim"       , int , default = 64    , aliases = ["hdim"])
    par.add_argument("model/n_layers"   , int , default = 1     , aliases = ["n_layers"])
    par.add_bool    ("model/no_bias" , aliases = ["no_bias"])
    with PreCondition(["model"], lambda m: m in ["MLP"]):
        par.add_bool    ("model/relu"   , aliases = ["relu"])


    par.add_argument("optimizer", str, default = "SGD", verify = lambda x: x in OPTIMIZERS)
    par.add_argument("optimizer/lr", float, default = 0.1   , aliases = ["lr"]) # learning rate
    par.add_argument("optimizer/wd", float, default = 0     , aliases = ["wd"]) # weight decay
    par.add_argument("optimizer/bs", float, default = 128   , aliases = ["bs"]) # batch size

    par.add_argument("n_epochs", int, default = 40) # num. of epochs

    return par.parse()

def get_dataaccess(name: str = ""):
    
    def myget(path: str):
        
        os.makedirs(os.path.dirname(path), exist_ok = True)

        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            c = pickle.loads( f.read() )
        return c

    def myset(path: str, content: bytes):
        os.makedirs(os.path.dirname(path), exist_ok = True)

        with open(path, "wb") as f:
            f.write(pickle.dumps(content))
        return True

    return  DataAccess(
        path = f".xingyun", 
        set_call = myset,
        get_call = myget,
    )
