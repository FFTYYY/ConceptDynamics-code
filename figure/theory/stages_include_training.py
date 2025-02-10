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

plt.rcParams.update({'font.size': 11})
norm = mcolors.Normalize(vmin=0, vmax=5)
colors = plt.get_cmap('Spectral_r')(np.linspace(0, 1, 256))
new_colors = np.vstack((colors[:100], colors[150:]))
cmap = LinearSegmentedColormap.from_list('no_yellow_spectral', new_colors)


if __name__ == "__main__":
    set_random_seed(2333)

    num_epochs = 2000
    eta = 0.01
    test_X = tc.FloatTensor([1,1.5,0])
    mu_1   = tc.FloatTensor([1,0,0])
    mu_2   = tc.FloatTensor([0,1.5,0])
    sigma = 0.05
    train_X_1 = tc.FloatTensor( np.random.multivariate_normal(mu_1, [[sigma,0,0],[0,sigma,0], [0,0,0]], 1000) )
    train_X_2 = tc.FloatTensor( np.random.multivariate_normal(mu_2, [[sigma,0,0],[0,sigma,0], [0,0,0]], 1000) )
    train_X   = tc.cat([train_X_1, train_X_2], dim = 0)

    d = len(test_X)

    U = tc.rand(d,d) / 5
    W = U @ U.t()

    A = tc.diag( (test_X ** 2) / 2 + sigma ** 2 )
    I = tc.eye(d, d)

    outputs = []
    outputs_train1 = []
    outputs_train2 = []
    losses = []
    losses_train = []
    Ws = []
    for idx_iter in range(num_epochs):

        update_W = W @ (A ** 2) + (A ** 2) @ W - W @ (A ** 2) @ W - ((A ** 2) @ (W @ W) + (W @ W) @ (A ** 2)) / 2
        W = W + eta * update_W

        loss   = 0.5 * tc.norm(W @ test_X - test_X) ** 2

        Ws.append(W)

        outputs.append(W @ test_X)
        losses.append(float(loss))

        outputs_train1.append(W @ mu_1)
        outputs_train2.append(W @ mu_2)

        loss_train = ( 0.5 * tc.norm(W @ train_X.t() - train_X.t(), dim = 0) ** 2 ).mean()
        losses_train.append(loss_train)
    
    Ws = tc.stack(Ws, dim = 0)
    outputs = tc.stack(outputs, dim = 0)
    outputs_train1 = tc.stack(outputs_train1, dim = 0)
    outputs_train2 = tc.stack(outputs_train2, dim = 0)
    losses       = tc.FloatTensor(losses)
    losses_train = tc.FloatTensor(losses_train)

    # ----------------------------- fig 2 curve -----------------------------
    if True:
        plt.figure(figsize = (4.6,4) , dpi = 256)
        # plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cmap(np.linspace(0, 1, cmap.N)))
        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.subplot()
        #plt.scatter( [0,1,0], [1.5,0,0] , s = 150 , label = "trainset" , marker = "o", color = "blue", facecolors="none")
        #plt.scatter( [1]   ,  [1.5]     , s = 150 , label = "testpoint", marker = "^", color = "orange")

        #plt.plot(outputs[:,0], outputs[:,1], color = cmap(0), label = "output")
        #plt.scatter( outputs[0][0], outputs[0][1], color = cmap(0), s = 10, lw=5, marker = "s")
        plt.scatter(  [0,1,0], [1.5,0,0], s = 300 , label = "Train" , marker = "o", lw = 1, facecolors="none", color=cmap(norm(0)), alpha=0.55)# , color = "blue", facecolors="none")
        plt.scatter( [1]   ,  [1.5]     , s = 300 , label = "Test", color = cmap(norm(0)), alpha=0.55) #, marker = "^", color = "orange")

        for idx , (outp , label) in enumerate( zip( [outputs, outputs_train1, outputs_train2], ["test", "train 1", "train 2"]) ):
            xx = outp[:,0]
            yy = outp[:,1]
            col = cmap(norm(idx + 1))
            plt.plot( xx, yy, lw=3, color = col, alpha=0.7, label = label)
            
            for pos in range(50, len(xx) - 1, 100):
                
                plt.annotate(
                    '', xy=(xx[pos + 1], yy[pos + 1]), xytext=(xx[pos], yy[pos]),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=0, alpha=0.7, mutation_scale=20)
                )

        plt.legend(loc='center right', frameon = True, borderpad=0.3, labelspacing=0.5, handlelength=1, markerscale=0.6, fontsize=11)
        plt.tight_layout()

        plt.xlim(-0.3,1.7)
        plt.ylim(-0.4,1.7)
        plt.yticks([0.0,0.5,1.0,1.5])
        plt.xticks([0.0,0.5,1.0,1.5])
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('grey')
            spine.set_linewidth(0.5)
        plt.savefig("theory_curve_w_train.pdf")
        plt.close()

    # ----------------------------- fig 3 entries -----------------------------
    if True:
        cmap_fuck = plt.get_cmap("Paired") 
        plt.style.use("default")
        plt.figure(figsize = (5.2,3.5) , dpi = 256)
        #plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cmap(np.linspace(0, 1, cmap.N)))
        fig = plt.subplot()
        line_1, = fig.plot(tc.arange(num_epochs), Ws[:,0,0], label = "$w_{1,1}$", color = cmap_fuck(0), lw=3, alpha=0.9)
        line_2, = fig.plot(tc.arange(num_epochs), Ws[:,1,1], label = "$w_{2,2}$", color = cmap_fuck(1), lw=3, alpha=0.9)
        line_3, = fig.plot(tc.arange(num_epochs), Ws[:,1,0], label = "$w_{2,1}$", color = cmap_fuck(2), lw=3, alpha=0.9)

        fig_2 = fig.twinx()
        line_4, = fig_2.plot(tc.arange(num_epochs), losses, "--", label = "Loss", color = "#FF95CA", lw=3, alpha=0.8)
        line_5, = fig_2.plot(tc.arange(num_epochs), losses_train, "--", label = "Train Loss", color = "#65AA90", lw=3, alpha=0.8)

        fig.annotate(
            "Init. Grow.", 
            xy       = (27, -18), 
            xytext   = (27, -33),
            fontsize = 6, 
            ha = "center", 
            va = "bottom", 
            xycoords="axes points", 
            arrowprops=dict(arrowstyle="-[, widthB=2, lengthB=.5", lw=1) , 
        )
        fig.annotate(
            "1st Supp.", 
            xy       = (59.5, -18), 
            xytext   = (59.5, -33),
            fontsize = 6, 
            ha = "center", 
            va = "bottom", 
            xycoords="axes points", 
            arrowprops=dict(arrowstyle="-[, widthB=2.0, lengthB=.5", lw=1) , 
        )
        fig.annotate(
            "2nd Grow.", 
            xy       = (172.5, -18), 
            xytext   = (172.5, -33),
            fontsize = 6, 
            ha = "center", 
            va = "bottom", 
            xycoords="axes points", 
            arrowprops=dict(arrowstyle="-[, widthB=15.4, lengthB=.5", lw=1) , 
        )
        fig.spines["top"].set_linewidth(0)
        fig.spines["bottom"].set_linewidth(2)
        fig.spines["right"].set_linewidth(0)
        fig.spines["left"].set_linewidth(2)
        #fig.tick_params(figis="both", which="both", bottom=True, top=False,
        #               labelbottom=True, left=True, right=False,
        #               labelleft=True, direction='out', length=5, width=1.0, pad=8, labelsize=17)
        
        fig_2.spines["top"].set_linewidth(0)
        fig_2.spines["bottom"].set_linewidth(0)
        fig_2.spines["right"].set_linewidth(2)
        fig_2.spines["left"].set_linewidth(0)

        lines = [line_1, line_2, line_3, line_4, line_5]
        labels = [line.get_label() for line in lines]
        fig.legend(lines, labels, frameon = True, borderpad=0.3, fontsize=12)

        fig.set_xlabel("Number of epochs")
        fig.xaxis.set_label_coords(0.5, -0.18)
        fig.set_ylabel("Value of $W$")
        fig_2.set_ylabel("Loss")
        plt.tight_layout()
        plt.savefig("theory_entries_and_loss_w_train.pdf")
        plt.close()






