import torch as tc
from xingyun import GlobalDataManager, MyDict
from paint.base import get_config, PROJECT
from tasks import get_data
from tasks.identity import map_2d as map_2d_identity
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from config import get_dataaccess

plt.rcParams.update({'font.size': 15})
cmap = plt.get_cmap("Paired")


def get_data(exp_id: str, device: str):
    cache_path = f"paint/cache/curve_noise_{exp_id}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            pred_Ys, dists = pickle.load(f)
    else:
        rec_G = GlobalDataManager(f"{PROJECT}/{exp_id}", data_access = get_dataaccess())
        rec_C = rec_G.get("config")
        
        if rec_C["data"] not in ["identity", "generative"]:
            raise NotImplementedError
        
        dists  = rec_C["dists"]

        _,_,_,_, info = rec_G.get_remote(f"saved_data/data.pkl")

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
            pickle.dump([pred_Ys, dists], f)

    return pred_Ys, dists


def paint(idx:int, exp_id: str, label: str, device: str):

    pred_Ys, dists = get_data(exp_id, device)
    
    #plt.plot([_[0] for _ in pred_Ys], [_[1] for _ in pred_Ys], color = cmap[idx], linewidth = 2, markersize = 10, marker = "+", label = f"output {label}")
    #plt.scatter( pred_Ys[0][0], pred_Ys[0][1], color = cmap[idx], s = 30, lw=5, marker = "+")
    color = cmap(idx)
    # line, = plt.plot([_[0] for _ in pred_Ys], [_[1] for _ in pred_Ys], color = color, linewidth = 4, marker=None, label = label, alpha=0.8)
    xx = [_[0] for _ in pred_Ys]
    yy = [_[1] for _ in pred_Ys]
    line, = plt.plot(xx, yy, color = color, linewidth = 3.5, marker=None, label = label, alpha=0.8)
    
    for pos in range(0, len(xx) - 1, 5):
        
        plt.annotate(
            '', xy=(xx[pos + 1], yy[pos + 1]), xytext=(xx[pos], yy[pos]),
            arrowprops=dict(arrowstyle='-|>', color=color, lw=0, alpha=0.6, mutation_scale=25)
        )

    # plt.scatter([_[0] for _ in pred_Ys][::2], [_[1] for _ in pred_Ys][::2], color = color, marker="<", s=90, alpha=0.7)

    # rd1,rd2,rd3 = float(tc.rand(1))*0.03,float(tc.rand(1))*0.02,-float(tc.rand(1))*0.02
    # plt.scatter( [dists[0],0,0+rd1], [0,dists[1]+rd2,0+rd3] , s = 150 , label = f"train set - {label}" , marker = "o", color = cmap(idx), facecolors="none")
    # plt.scatter( [dists[0]]   ,  [dists[1]]     , s = 150 , label = f"test point - {label}", marker = "^", color = cmap(idx))

    return dists, line

def main(C: MyDict):

    exp_ids = C["exp_id"].strip().split(",")
    labels  = C["info"].strip().split(",")
    labels  = ["${\\bf\\mu}=(%s,2)$" % l for l in labels]
 
    device  = C["device"]

    fig = plt.figure(figsize = (6,6), dpi = 256)
    plt.style.use("seaborn-v0_8-whitegrid")

    plots = {}
    all_dists = []
    for idx, (exp_id, label) in enumerate( zip(exp_ids, labels) ):
        dists, line = paint(idx, exp_id, label, device)
        all_dists.append( dists )
        plots[label] = {"line": line}


    for idx in range(len(exp_ids)):
        dists = all_dists[idx]
        label = labels[idx]
        rd1,rd2,rd3 = float(tc.rand(1))*0.03,float(tc.rand(1))*0.02,-float(tc.rand(1))*0.02
        train_plot = plt.scatter( [dists[0],0,0+rd1], [0,dists[1]+rd2,0+rd3] , s = 300 , marker = "o", edgecolors=cmap(idx), facecolors="none", lw=1.2, alpha=0.9) #, facecolors="none")

        plots[label]["train"] = train_plot

    for idx in range(len(exp_ids)):
        dists = all_dists[idx]
        label = labels[idx]
        test_plot = plt.scatter( [dists[0]],  [dists[1]], s = 300 , marker = "o", color = cmap(idx), alpha=0.6)
        plots[label]["test"] = test_plot

    spacing_marker = plt.scatter([], [], color='white', alpha=0, label='spacing')
    legends_markers = []
    legend_labels = []
    for label in plots:
        line, train_plot, test_plot = plots[label]["line"], plots[label]["train"], plots[label]["test"]
        legends_markers.append( (line, train_plot, test_plot) )
        legend_labels  .append( label + " Trajectory / Train / Test" )



    legend = plt.legend(
        legends_markers, 
        legend_labels, 
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper right", 
        frameon = True, 
        # borderpad=0.3, 
        #  labelspacing=0.3, 
        handlelength=4, 
        handletextpad=1.5 , 
        markerscale=0.6, 
        # borderaxespad=0.1
    )
    plt.tight_layout()

    #plt.xlim(-0.5,3.5)
    #plt.ylim(-0.5,2.5)
    plt.xticks([0.0, 1.0, 2.0, 3.0], ["0.0", "1.0", "2.0", "3.0"],fontsize=20)
    plt.yticks([0.0, 1.0, 2.0, 3.0], ["0.0", "1.0", "2.0", "3.0"],fontsize=20)
    plt.xlim(-0.4,max(dists[0], dists[1]) + 0.4)
    plt.ylim(-0.4,max(dists[0], dists[1]) + 0.4)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('grey')
        spine.set_linewidth(0.5)
    # plt.show()
    # plt.title(f"{rec_C['group']}/{rec_C['info']}")
    prefix = '' if C['folder_prefix'] == '' else C['folder_prefix'] + '/'
    plt.savefig(f"figure/{prefix}curve/dist-{','.join(exp_ids)}.pdf")    

if __name__ == "__main__":

    C       = get_config()
    ##logger  = Logger()
    ##my_id   = get_id(f"{PROJECT}/analysis")
    ##logger.log("start.")

    exp_id = C["exp_id"]
    
    ##logger.log("initialized.")
    ##logger.log(f"config     = {pformat(dict(C))}")
    # G.set("logger", logger)

    main(C)

    ##logger.log("finished.")




