'''this model probes the order of task learned'''

import torch as tc
from xingyun import GlobalDataManager, MyDict, Logger
from paint.base import get_config, PROJECT
from pprint import pformat
from tasks import get_data
from tasks.multi_identity import map_lowdim 
from matplotlib import pyplot as plt
import pickle
from itertools import product   
import networkx as nx
from matplotlib import colormaps
import os
from config import get_dataaccess


def make_nodes(num_class):
    '''each node represents a class, and the value of the node is the class label.'''
    nodes = product(* [range(2) for _ in range(num_class)])
    return [tc.FloatTensor(_n) for _n in nodes]

def combinatory_number(n: int, m:int):
    return int( tc.prod(tc.arange(n-m+1, n+1))) // int(tc.prod(tc.arange(1, m+1)) )

def get_data(exp_id: str, device: str):
    cache_path = f"paint/cache/topology_order_{exp_id}.pkl"


    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            info, num_class, dists, checkpoints, models = pickle.load(f)
    else:

        rec_G = GlobalDataManager(f"{PROJECT}/{exp_id}", data_access = get_dataaccess()) # recover from the training 
        rec_C = rec_G.get("config")
        
        # --- recover config ---
        dim         = rec_C["dim"]
        num_class   = rec_C["num_class"]
        num_sample  = rec_C["num_sample"]
        sigma       = rec_C["sigma"]
        noise       = rec_C["noise"]
        dists       = rec_C.get("dists_multi") or rec_C.get("dists")

        assert rec_C["data"] == "multi_identity"
        assert dists is not None


        # --- recover data ---
        # data_seed = rec_G.get("data_seed")    
        # with FixRandom(data_seed):
        train_X, train_Y, test_X, test_Y, info = rec_G.get_remote(f"saved_data/data.pkl")
        
        checkpoints = rec_G.summarize("checkpoints", lambda dic_check:[dic_check[_] for _ in dic_check])
        models = [rec_G.get_remote(f"saved_checkpoints/{checkpoint}.pkl").to(device) for checkpoint in checkpoints]

        with open(cache_path, "wb") as f:
            pickle.dump([info, num_class, dists, checkpoints, models], f)

    return info, num_class, dists, checkpoints, models


def main(C: MyDict):
    exp_id = C["exp_id"]
    device = C["device"]
    info, num_class, dists, checkpoints, models = get_data(exp_id, device)

    U = info["U"]
    Uupdim = U[:num_class]

    
    mus = tc.eye(num_class) * tc.FloatTensor(dists).view(-1)
    nodes = make_nodes(num_class)

    # --- make test data ---
    my_test_X = tc.cat([(mus * node).sum(dim = 0).view(1,-1) for node in  nodes], dim = 0) # (num_nodes, num_class)
    my_test_Y = my_test_X.clone().detach()

    my_test_X = my_test_X @ Uupdim # (num_nodes, dim)

    # --- get graph ---
    edges = []
    for (n1_idx, n1), (n2_idx, n2) in product(enumerate(nodes), enumerate(nodes)):
        if 0.95 <= float( (n1 - n2).abs().sum() ) <= 1.05 and n1_idx < n2_idx:
            edges.append((n1_idx, n2_idx))
    pos = {}
    pos_xs = {}
    for node_idx, node in enumerate(nodes):
        node_y = int(node.sum())
        if pos_xs.get(node_y) is None:
            pos_xs[node_y] = 0
        offset = (combinatory_number(num_class, node_y) + 1 ) / 2
        pos[node_idx] = [ node_y, float(pos_xs[node_y]) - offset ]
        pos_xs[node_y] = pos_xs[node_y] + 1

    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(edges)


    # --- recover model ---
    cmap   = colormaps["YlGnBu"]
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 0, vmax = 1))
    fig = plt.figure(figsize = (5,5))
    ax = fig.subplots()
    plt.colorbar(sm, ax=ax, label="Loss Value")  # draw a colorbar
    plt.savefig(f"figure/topology_order/colorbar.png")
    plt.close()

    images = []
    for (model, checkpoint) in zip(models, checkpoints):
        if checkpoint not in [0,2,4]:
            continue
        model = model.to(device)

        with tc.no_grad():
            model = model.eval()
            pred_Y = model(my_test_X.to(device)) # (num_nodes, dim)
        _, m_pred_Y = map_lowdim(None , pred_Y, num_class, info, device)

        losses = []
        for node_idx, node in enumerate(nodes):
            loss = (tc.norm(m_pred_Y[node_idx].to(device) - my_test_Y[node_idx].to(device)) ** 2 ) / 2
            # print (f"node = {node} pred = {m_pred_Y[node_idx]}, loss={loss:.4f}")
            losses.append(float(loss))

        # --- draw the results ---

        # normalized_loss = [ max( min(r / 1 , 0.5), 0.0) for r in losses]
        colors   = [ cmap(r)   for r in losses ]


        fig = plt.figure(figsize = (5,5))
        ax = fig.subplots()
        # plt.title(f"{rec_C['group']}/{rec_C['info']}/{dists}/{checkpoint}")
        # plt.colorbar(sm, ax=ax, label="Loss Value")  # 为颜色条添加标签

        # pos = nx.spectral_layout(graph)  
        nx.draw(graph, pos, node_size=1500, node_color=colors, ax=ax)

        # Optional: Customize the edge labels
        nx.draw_networkx_edges(graph, pos=pos, edgelist=edges, edge_color = "black")

        for cur_nid, cur_nd in enumerate(nodes):
            node_labels = {node_idx: ", ".join([str(int(_)) for _ in node]) if node_idx == cur_nid else "" for node_idx, node in enumerate(nodes)}
            color = "white" if losses[cur_nid] > 0.6 else "black"
            nx.draw_networkx_labels(graph, pos, labels = node_labels, font_size=8, font_color = color)
        plt.tight_layout()
        
        prefix = '' if C['folder_prefix'] == '' else C['folder_prefix'] + '/'
        plt.savefig(f"figure/{prefix}topology_order/epoch_{checkpoint + 1}.png")
        
if __name__ == "__main__":

    C       = get_config()
    logger  = Logger()
    logger.log("start.")

    logger.log("initialized.")
    logger.log(f"config     = {pformat(dict(C))}")
    # G.set("logger", logger)

    main(C)

    logger.log("finished.")




