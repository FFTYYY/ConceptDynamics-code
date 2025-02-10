import torch as tc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
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

num_epochs = 10000
eta = 0.01
test_X = tc.FloatTensor([0.7,1.7,3])
sigma = tc.FloatTensor([0.05,0.05,0.05])
d = len(test_X)

def plot_lines(ax, point_x, point_y, point_z):
    ax.plot([point_x, point_x], [point_y, 0], [point_z, point_z], linestyle="dashed", color="gray")
    ax.plot([point_x, 0], [point_y, point_y], [point_z, point_z], linestyle="dashed", color="gray")
    ax.plot([point_x, point_x], [point_y, point_y], [point_z, 0], linestyle="dashed", color="gray")

def copy(x: list, k: int):
    return x if k <= 1 else copy(x + x, k // 2)


def get_trace():
    set_random_seed(233333)
    plt.style.use("default")


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

    A = tc.diag( (test_X ** 2) / d + sigma ** 2 )
    I = tc.eye(d, d)

    outputs = []
    losses = []
    Ws = []
    for idx_iter in range(num_epochs):

        update_W = W @ (A ** 2) + (A ** 2) @ W - W @ (A ** 2) @ W - ((A ** 2) @ (W @ W) + (W @ W) @ (A ** 2)) / 2
        W = W + eta * update_W

        loss   = 0.5 * tc.norm(W @ test_X - test_X) ** 2

        Ws.append(W)
        outputs.append(W @ test_X)
        losses.append(float(loss))

    
    Ws = tc.stack(Ws, dim = 0)
    outputs = tc.stack(outputs, dim = 0)
    losses = tc.FloatTensor(losses)

    return Ws, outputs, losses

def draw_trajectory(outputs):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    mu = test_X

    xs = tc.stack( [ tc.randn(1000) * 0.05 + mu[0], tc.randn(1000) * 0.05, tc.randn(1000) * 0.05, tc.randn(1000) * 0.05], dim = 0).view(-1)
    ys = tc.stack( [ tc.randn(1000) * 0.05, tc.randn(1000) * 0.05 + mu[1], tc.randn(1000) * 0.05, tc.randn(1000) * 0.05], dim = 0).view(-1)
    zs = tc.stack( [ tc.randn(1000) * 0.05, tc.randn(1000) * 0.05, tc.randn(1000) * 0.05 + mu[2], tc.randn(1000) * 0.05], dim = 0).view(-1)

    ax.scatter(xs, ys, zs, label = "Train points", color = cmap(norm(2)))
    
    xs_test = copy([0        , mu[0], mu[0], mu[0], ]  , 8)
    ys_test = copy([mu[1], mu[1], 0        , mu[1], ]  , 8)
    zs_test = copy([mu[2], 0        , mu[2], mu[2], ]  , 8)
    ax.scatter(xs_test[-1], ys_test[-1], zs_test[-1], marker = "^", s = 200, label = "Test point", color = cmap(norm(1)))


    for k in range(len(xs_test)):
        plot_lines(ax, xs_test[k], ys_test[k], zs_test[k])
    plot_lines(ax, 0,0,0)
    plot_lines(ax, mu[0],0,0)
    plot_lines(ax, 0,mu[1],0)
    plot_lines(ax, 0,0,mu[2])

    arrow_length = 0.1  # Set a length for the arrows
    for i in range(0, len(outputs) - 100, 5):  # Step through points every 50
        rat = float( (outputs[1] - outputs[0]).norm() / (outputs[i+1] - outputs[i]).norm())
        ax.quiver(
            outputs[i, 0], outputs[i, 1], outputs[i, 2],  # Starting point
            outputs[i+1, 0] - outputs[i, 0],  # X direction
            outputs[i+1, 1] - outputs[i, 1],  # Y direction
            outputs[i+1, 2] - outputs[i, 2],  # Z direction
            length=arrow_length, color=cmap(norm(4)), arrow_length_ratio= 40 * rat 
        )

    ax.plot(outputs[:,0], outputs[:,1], outputs[:,2], color = cmap(norm(4)), lw = 2)

    plt.tight_layout()
    # ax.set_xlim([-0.2,3])
    # ax.set_ylim([-0.2,3])
    # ax.set_zlim([-0.2,3])

    plt.legend()
    # plt.show()
    plt.savefig("fail_3d.pdf")


def draw_loss(outputs):
    plt.figure(figsize = (4.6,4) , dpi = 256)
    # plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cmap(np.linspace(0, 1, cmap.N)))
    fig = plt.subplot()
    line_1, = fig.plot(tc.arange(num_epochs), Ws[:,0,0], label = "$w_{1,1}$", color = cmap(norm(1)), lw=2, alpha=0.7)
    line_2, = fig.plot(tc.arange(num_epochs), Ws[:,1,1], label = "$w_{2,2}$", color = cmap(norm(2)), lw=2, alpha=0.7)
    line_3, = fig.plot(tc.arange(num_epochs), Ws[:,2,2], label = "$w_{3,3}$", color = cmap(norm(3)), lw=2, alpha=0.7)
    line_4, = fig.plot(tc.arange(num_epochs), Ws[:,0,1], label = "$w_{1,2}$", color = cmap(norm(4)), lw=2, alpha=0.7)
    line_5, = fig.plot(tc.arange(num_epochs), Ws[:,0,2], label = "$w_{1,3}$", color = cmap(norm(5)), lw=2, alpha=0.7)
    line_6, = fig.plot(tc.arange(num_epochs), Ws[:,1,2], label = "$w_{2,3}$", color = cmap(norm(6)), lw=2, alpha=0.7)

    fig_2 = fig.twinx()
    line_0, = fig_2.plot(tc.arange(num_epochs), losses, "--" , label = "Loss", color = cmap(norm(0)), lw=2, alpha=1)

    fig.spines["top"].set_linewidth(0)
    fig.spines["bottom"].set_linewidth(2)
    fig.spines["right"].set_linewidth(0)
    fig.spines["left"].set_linewidth(2)
    fig_2.spines["top"].set_linewidth(0)
    fig_2.spines["bottom"].set_linewidth(2)
    fig_2.spines["right"].set_linewidth(0)
    fig_2.spines["left"].set_linewidth(2)

    lines = [line_0, line_1, line_2, line_3, line_4, line_5, line_6]
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
    plt.savefig("fail.pdf")
    plt.close()

if __name__ == "__main__":
    # ----- plot the curve and quivers ----- 
    Ws, outputs, losses = get_trace()
    
    draw_loss(outputs)
    draw_trajectory(outputs)