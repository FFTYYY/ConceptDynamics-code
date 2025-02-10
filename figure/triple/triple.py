import torch as tc
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
import matplotlib.transforms as mtrans
from xingyun import set_random_seed
import numpy as np
import seaborn as sns

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

cmap = sns.color_palette("cool", 7)
plt.rcParams.update({'font.size': 11})
norm = mcolors.Normalize(vmin=0, vmax=6)
colors = plt.get_cmap('Spectral_r')(np.linspace(0, 1, 256))
new_colors = np.vstack((colors[:100], colors[150:]))
cmap = LinearSegmentedColormap.from_list('no_yellow_spectral', new_colors)

if __name__ == "__main__":
    set_random_seed(233333)
    plt.style.use("default")

    num_epochs = 10000
    eta = 0.01
    test_X = tc.FloatTensor([1,1.7,2.25])
    sigma = tc.FloatTensor([0.5,0.25,0.25])
    d = len(test_X)
    mu = test_X

    U = tc.rand(d,d) / 10
    W = U @ U.t()
    # W = tc.FloatTensor([
    #     [0.0044, 0.0070, 0.0023] , 
    #     [0.0070, 0.0104, 0.0058] , 
    #     [0.0023, 0.0058, 0.0022] , 
    # ])
    # _U,_S,_V = W.svd()
    # U = _U @ (_S ** 0.5).diag()
    # W = U @ U.t()
    print (W)
    
    xs = tc.stack( [ tc.randn(1000) * sigma[0] + mu[0], tc.randn(1000) * sigma[0], tc.randn(1000) * sigma[0], tc.randn(1000) * sigma[0]], dim = 0).view(-1)
    ys = tc.stack( [ tc.randn(1000) * sigma[1], tc.randn(1000) * sigma[1] + mu[1], tc.randn(1000) * sigma[1], tc.randn(1000) * sigma[1]], dim = 0).view(-1)
    zs = tc.stack( [ tc.randn(1000) * sigma[2], tc.randn(1000) * sigma[2], tc.randn(1000) * sigma[2] + mu[2], tc.randn(1000) * sigma[2]], dim = 0).view(-1)
    train_X = tc.stack([xs,ys,zs], dim = 1)

    A = tc.diag( (test_X ** 2) / d + sigma ** 2 )
    I = tc.eye(d, d)

    outputs = []
    losses = []
    train_losses = []
    Ws = []
    for idx_iter in range(num_epochs):

        update_W = W @ (A ** 2) + (A ** 2) @ W - W @ (A ** 2) @ W - ((A ** 2) @ (W @ W) + (W @ W) @ (A ** 2)) / 2
        W = W + eta * update_W

        loss       = 0.5 * tc.norm(W @ test_X - test_X) ** 2
        train_loss = 0.5 * ((train_X @ W.t() - train_X) ** 2).sum(dim = -1).mean()

        Ws.append(W)
        outputs.append(W @ test_X)
        losses.append(float(loss))
        train_losses.append(float(train_loss))

    
    Ws = tc.stack(Ws, dim = 0)
    outputs = tc.stack(outputs, dim = 0)
    losses = tc.FloatTensor(losses)
    train_losses = tc.FloatTensor(train_losses)

    import numpy as np
    np.save("train_losses.npy" , train_losses.numpy())

    plt.figure(figsize = (4.6,4) , dpi = 256)
    # plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cmap(np.linspace(0, 1, cmap.N)))
    fig = plt.subplot()
    line_1, = fig.plot(tc.arange(num_epochs), Ws[:,0,0], label = "$w_{1,1}$", color = cmap(norm(2)), lw=2, alpha=0.7)
    line_2, = fig.plot(tc.arange(num_epochs), Ws[:,1,1], label = "$w_{2,2}$", color = cmap(norm(3)), lw=2, alpha=0.7)
    line_3, = fig.plot(tc.arange(num_epochs), Ws[:,2,2], label = "$w_{3,3}$", color = cmap(norm(4)), lw=2, alpha=0.7)
    line_4, = fig.plot(tc.arange(num_epochs), Ws[:,0,1], label = "$w_{1,2}$", color = cmap(norm(5)), lw=2, alpha=0.7)
    line_5, = fig.plot(tc.arange(num_epochs), Ws[:,0,2], label = "$w_{1,3}$", color = cmap(norm(6)), lw=2, alpha=0.7)
    line_6, = fig.plot(tc.arange(num_epochs), Ws[:,1,2], label = "$w_{2,3}$", color = cmap(norm(7)), lw=2, alpha=0.7)

    fig_2 = fig.twinx()
    line_0, = fig_2.plot(tc.arange(num_epochs), losses, "--" , label = "Loss"             , color = cmap(norm(0)), lw=2, alpha=1)
    line_m1, = fig_2.plot(tc.arange(num_epochs), train_losses, "--" , label = "Train Loss", color = cmap(norm(1)), lw=2, alpha=1)

    fig.spines["top"].set_linewidth(0)
    fig.spines["bottom"].set_linewidth(2)
    fig.spines["right"].set_linewidth(0)
    fig.spines["left"].set_linewidth(2)
    fig_2.spines["top"].set_linewidth(0)
    fig_2.spines["bottom"].set_linewidth(2)
    fig_2.spines["right"].set_linewidth(0)
    fig_2.spines["left"].set_linewidth(2)

    lines = [line_0, line_1, line_2, line_3, line_4, line_5, line_6, line_m1]
    labels = [line.get_label() for line in lines]
    # fig.legend(lines, labels, frameon = 1)
    fig.legend(lines, labels, frameon = True, fontsize=11, loc = "center left")

    fig.set_xlabel("Number of epochs")
    # fig.xaxis.set_label_coords(0.5, -0.15)
    fig.set_ylabel("Value of $W$")
    fig_2.set_ylabel("Loss")
    plt.xscale("log")
    plt.tight_layout()
    # plt.show()
    plt.savefig("triple.pdf")
    plt.close()






