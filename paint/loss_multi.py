import torch as tc
from paint.base import get_config, PROJECT
from xingyun import GlobalDataManager, MyDict, Logger
from pprint import pformat
from tasks import get_data
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm 
import os
import matplotlib.colors as mcolors
import numpy as np
from config import get_dataaccess
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.size": 20})
# cmap = sns.color_palette("cool", 6)

# colors = {0: "#56C1FF", 1: "#FF95CA", 2: "#ED220D", 3: "#ff42a1"}
norm = mcolors.Normalize(vmin=0, vmax=5)
colors = plt.get_cmap('Spectral_r')(np.linspace(0, 1, 256))
new_colors = np.vstack((colors[:100], colors[150:]))
cmap = LinearSegmentedColormap.from_list('no_yellow_spectral', new_colors)

def get_data(exp_id: str, device: str):
    cache_path = f"paint/cache/loss_multi_{exp_id}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            checkpoints, losses, dists = pickle.load(f)
    else:

        rec_G = GlobalDataManager(f"{PROJECT}/{exp_id}", data_access = get_dataaccess())
        rec_C = rec_G.get("config")
        
        if rec_C["data"] not in ["identity", "generative"]:
            raise NotImplementedError
        
        dists  = rec_C["dists"]

        _,_,_,_, info = rec_G.get_remote(f"saved_data/data.pkl")

        my_test_X = tc.FloatTensor([dists[0], dists[1]]).view(1,-1).to(device) @ info["U"].to(device)

        losses = []
        pred_Ys = []
        checkpoints = rec_G.summarize("checkpoints", lambda dic_check:[dic_check[_] for _ in dic_check])
        checkpoints = checkpoints[:50]
        for epoch_idx in tqdm(checkpoints):
            model = rec_G.get_remote(f"saved_checkpoints/{epoch_idx}.pkl")

            model = model.to(device)
            model = model.eval()

            pred_Y = model(my_test_X).detach().cpu()[0]
            loss = ((pred_Y - my_test_X).norm() ** 2) / 2

            losses.append(float(loss))
        
        with open(cache_path, "wb") as f:
            pickle.dump([checkpoints, losses, dists], f)

    return checkpoints, losses, dists


def paint(idx:int, exp_id: str, label: str, device: str):
    
    checkpoints, losses, dists = get_data(exp_id, device)

    #plt.plot(checkpoints, losses, label = label, color = colors[idx])
    plt.plot([c + 1 for c in checkpoints], losses, label = label, color=cmap(norm(idx + 1)), lw = 3, alpha=0.7) #color = cmap(idx))

    return dists

def main(C: MyDict):
    global norm

    exp_ids = C["exp_id"].strip().split(",")
    labels  = C["info"].strip().split(",")
    device  = C["device"]
    norm = mcolors.Normalize(vmin=0, vmax= max( len(exp_ids) + 1, 3) )

    fig = plt.figure(figsize = (6,3), dpi=256)

    dists = None
    idx = 0
    for idx, (exp_id, label) in enumerate( zip(exp_ids, labels) ):
        try:
            float(label)
            label = "$\sigma={%s},.05$" % label
        except ValueError:
            pass
        now_dists = paint(idx, exp_id, label, device)
        if dists is None:
            dists = now_dists
        else:
            pass

    plt.ylabel("Test loss", fontsize=19)
    plt.xlabel("Training time", fontsize=19)
    legend = plt.legend(frameon = 1, fontsize=16)
    # frame.set_facecolor((0.9,0.9,0.9,0.6))
    ax = plt.gca()
    ax.spines["top"].set_linewidth(0)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(axis="both", which="both", bottom=True, top=False,
                   labelbottom=True, left=True, right=False,
                   labelleft=True,direction='out',length=5,width=1.0,pad=8,labelsize=16)
    plt.tight_layout()

    # plt.show()
    # plt.title(f"{rec_C['group']}/{rec_C['info']}")
    prefix = '' if C['folder_prefix'] == '' else C['folder_prefix'] + '/'
    plt.savefig(f"figure/{prefix}curve/loss-{','.join(exp_ids)}.pdf", bbox_inches='tight', pad_inches=0)
    plt.close()   

if __name__ == "__main__":

    C       = get_config()
    logger  = Logger()
    logger.log("start.")
    
    logger.log("initialized.")
    logger.log(f"config     = {pformat(dict(C))}")
    # G.set("logger", logger)

    main(C)

    logger.log("finished.")




