from xingyun import set_random_seed, GlobalDataManager, Logger, get_id, make_hook_logger, FixRandom
from config import get_config, get_dataaccess, PROJECT
from pprint import pformat
from tasks import get_data
from models import get_model
from train_and_test import get_optimizer, train, test
from random import randint
import copy
import pickle
import torch as tc

def main(G: GlobalDataManager):
    C = G.get("config")
    device = C["device"]

    train_X, train_Y, test_X, test_Y, info = get_data(C["data"], C("data"))
    G.set_remote(f"saved_data/data.pkl", [train_X, train_Y, test_X, test_Y, info])

    model = get_model(C["model"], train_X.size(1), train_Y.size(1), C["hdim"], C["n_layers"], C("model"))
    optimizer = get_optimizer(C["optimizer"], C("optimizer"), model)

    train_X, train_Y, test_X, test_Y = map(lambda x: x.to(device), [train_X, train_Y, test_X, test_Y])
    model = model.to(device)
    
    bs = C["bs"]
    cur_step = 0
    for epoch_idx in range(C["n_epochs"]):
        G.set_remote(f"saved_checkpoints/{epoch_idx}.pkl", copy.deepcopy(model).cpu().eval())
        G.set("checkpoints", epoch_idx)
        
        for batch_start in range(0, len(train_X), bs):
            cur_step = cur_step + 1
            batch_X = train_X[batch_start:batch_start + bs]
            batch_Y = train_Y[batch_start:batch_start + bs]

            test_loss = test(model, test_X, test_Y)
            train_loss, model = train(model, optimizer, batch_X, batch_Y)

            G.set("train_loss", train_loss)
            G.set("test_loss" , test_loss)
        logger.log(f"epoch {epoch_idx} finished. train_loss = {train_loss: .4f}, test_loss = {test_loss: .4f}")

    return model


if __name__ == "__main__":

    C       = get_config()
    logger  = Logger()
    my_id   = int(C["id"])
    logger.log("start.")

    G = GlobalDataManager(
        name = f"{PROJECT}/{my_id}" , 
        hooks = [make_hook_logger(logger)] , 
        data_access = get_dataaccess() , 
    )

    logger.log("initialized.")
    logger.log(f"PROJECT    = {PROJECT}")
    logger.log(f"my_id      = {my_id}")
    logger.log(f"config     = {pformat(dict(C))}")
    G.set("config", C)

    if C["seed"] >= 0:
        set_random_seed(C["seed"])
        logger.log(f"random seed set to {C['seed']}")

    logger.log("enter main.")
    logger.log("-" * 50)

    main(G)

    logger.log("-" * 50)
    logger.log("exit main.")

    G.upload_data()
    logger.log("finished.")




