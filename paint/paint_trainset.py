import torch as tc
from xingyun import GlobalDataManager, MyDict, Logger
from paint.base import get_config, PROJECT
from pprint import pformat
from tasks import get_data
from tasks.identity import map_2d as map_2d_identity
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm 
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from config import get_dataaccess


plt.rcParams.update({"font.size": 20})
norm = mcolors.Normalize(vmin=0, vmax=5)
colors = plt.get_cmap('Spectral_r')(np.linspace(0, 1, 256))
new_colors = np.vstack((colors[:100], colors[150:]))
cmap = LinearSegmentedColormap.from_list('no_yellow_spectral', new_colors)


scale_factor = 78
def rescaled_conv(arr, N):
    conv_result = np.convolve(arr, np.ones(N) / N, mode='full')
    conv_result[:N-1] = conv_result[:N-1] * N/np.arange(1,N)
    return conv_result[0: len(conv_result)-N+1]


def get_data(exp_id: str, device: str):
    cache_path = f"paint/cache/paint_trainset_{exp_id}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            pred_Ys, dists, dataset = pickle.load(f)

    else:
        rec_G = GlobalDataManager(f"{PROJECT}/{exp_id}", data_access = get_dataaccess())
        rec_C = rec_G.get("config")
        if rec_C["data"] not in ["identity", "generative", "identity_truncate"]:
            raise NotImplementedError
        dists  = rec_C["dists"]
        train_X, train_Y, test_X, test_Y, info = rec_G.get_remote(f"saved_data/data.pkl")
        dataset = [train_X.cpu(), train_Y.cpu(), test_X.cpu(), test_Y.cpu(), info]

        my_test_X = tc.FloatTensor([dists[0], dists[1]]).view(1,-1).to(device) @ info["U"].to(device)
        pred_Ys = []
        checkpoints = rec_G.summarize("checkpoints", lambda dic_check:[dic_check[_] for _ in dic_check])
        for epoch_idx in tqdm(checkpoints):
            model = rec_G.get_remote(f"saved_checkpoints/{epoch_idx}.pkl")

            model = model.to(device)
            model = model.eval()

            pred_Y = model(my_test_X).detach().cpu()[0]
            _, m_pred_Y = map_2d_identity(None , pred_Y, info, device)

            pred_Ys.append(m_pred_Y)
        
        with open(cache_path, "wb") as f:
            pickle.dump([pred_Ys, dists, dataset], f)

    return pred_Ys, dists, dataset

_dataset_flag = False
def paint(idx:int, exp_id: str, label: str, device: str):
    global _dataset_flag

    pred_Ys, dists, dataset = get_data(exp_id, device)

    color = cmap(norm(idx + 1))

    xx = np.array([_.cpu()[0] for _ in pred_Ys])
    yy = np.array([_.cpu()[1] for _ in pred_Ys])
    plt.plot(xx, yy, color = color, linewidth = 3.5, marker=None, alpha=0.7, label=label)
    for pos in range(0, len(xx) - 1, 2):
        
        plt.annotate(
            '', xy=(xx[pos + 1], yy[pos + 1]), xytext=(xx[pos], yy[pos]),
            arrowprops=dict(arrowstyle='-|>', color=color, lw=0, alpha=0.85, mutation_scale=25)
        )

    if _dataset_flag == False:
        _dataset_flag = True
    
        train_X, train_Y, test_X, test_Y, info = dataset
        train_X, train_Y = map_2d_identity(train_X, train_Y, info, device)
        plt.scatter(train_X.cpu()[:,0], train_X.cpu()[:,1], s = 2, color = "#9050F0", alpha=0.6, label = "Train")


    return dists

def main(C: MyDict):
    global norm 

    exp_ids = C["exp_id"].strip().split(",")
    labels  = C["info"].strip().split(",") if C["info"] != "" else [None for _ in exp_ids]
    device  = C["device"]

    norm = mcolors.Normalize(vmin=0, vmax= max( len(exp_ids) + 1, 3) )

    fig = plt.figure(figsize = (6,6), dpi=256)
    plt.style.use("seaborn-v0_8-whitegrid")

    dists = None
    idx = 0
    for idx, (exp_id, label) in enumerate( zip(exp_ids, labels) ):
        if label is not None:
            try:
                float(label)
                label = "${\\bf\\sigma}=("+str(label)+",0.05)$" #"$\sigma={%s},.05$" % label
            except ValueError:
                pass
        now_dists = paint(idx, exp_id, label, device)
        if dists is None:
            dists = now_dists
        else:
            assert str(dists) == str(now_dists)

    color = cmap(norm(0))
    # plt.scatter( [dists[0],0,0], [0,dists[1],0], s = 350 , label = f"Train", marker = "o", edgecolors=color, facecolors="none", lw=1.2, alpha=0.6)
    plt.scatter( [dists[0]]   ,  [dists[1]]    , s = 350 , label = f"Test", marker = "o", color = color, alpha=0.6) #cmap(idx)
    legend = plt.legend(loc="center right", frameon = True, borderpad=0.3, labelspacing=0.6, handlelength=1.5, markerscale=0.8, bbox_to_anchor=(1.0,0.4), fontsize=16)
    # frame.set_facecolor((0.9,0.9,0.9,0.6))

    xmax = max(dists[0],dists[1]) 
    plt.xlim(-0.3,xmax + 0.2)
    plt.ylim(-0.3,xmax + 0.2)
    if xmax == 2.0: 
        plt.yticks([0.0, 1.0, 2.0], ["0.0", "1.0", "2.0"],fontsize=20)
        plt.xticks([0.0, 1.0, 2.0], ["0.0", "1.0", "2.0"],fontsize=20)
    else: 
        plt.xticks([0.0, 2.0, 4.0], ["0.0", "2.0", "4.0"],fontsize=20)
        plt.yticks([0.0, 2.0, 4.0], ["0.0", "2.0", "4.0"],fontsize=20)

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('grey')
        spine.set_linewidth(0.5)
    plt.tight_layout()
    # plt.show()
    # plt.title(f"{rec_C['group']}/{rec_C['info']}")
    prefix = '' if C['folder_prefix'] == '' else C['folder_prefix'] + '/'
    out_filename = f"figure/{prefix}rebuttal/dataset_{','.join(exp_ids)}.png"
    print("## save to", out_filename)
    plt.savefig(out_filename)    

if __name__ == "__main__":

    C       = get_config()
    logger  = Logger()
    logger.log("start.")
    
    logger.log("initialized.")
    logger.log(f"config     = {pformat(dict(C))}")
    # G.set("logger", logger)

    main(C)

    logger.log("finished.")




