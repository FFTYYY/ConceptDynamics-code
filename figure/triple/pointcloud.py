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
test_X = tc.FloatTensor([1,1.7,2.25])
sigma = tc.FloatTensor([0.5,0.25,0.25])
d = len(test_X)

def plot_lines(point_x, point_y, point_z):
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


if __name__ == "__main__":
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = tc.stack( [ tc.randn(1000) * 0.05 + test_X[0], tc.randn(1000) * 0.05, tc.randn(1000) * 0.05], dim = 0).view(-1)
    ys = tc.stack( [ tc.randn(1000) * 0.05, tc.randn(1000) * 0.05 + test_X[1], tc.randn(1000) * 0.05], dim = 0).view(-1)
    zs = tc.stack( [ tc.randn(1000) * 0.05, tc.randn(1000) * 0.05, tc.randn(1000) * 0.05 + test_X[2]], dim = 0).view(-1)

    ax.scatter(xs, ys, zs, label = "train set")
    
    xs_test = copy([0        , test_X[0], test_X[0], test_X[0], ]  , 8)
    ys_test = copy([test_X[1], test_X[1], 0        , test_X[1], ]  , 8)
    zs_test = copy([test_X[2], 0        , test_X[2], test_X[2], ]  , 8)
    ax.scatter(xs_test, ys_test, zs_test, marker = "^", s = 200, label = "test points")




    arrow_length = 1.3
    ax.quiver(0, 0, 0, arrow_length * test_X[0], 0           , 0            , color=(.2,.2,.2), arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0           , arrow_length * test_X[1], 0            , color=(.2,.2,.2), arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0           , 0           , arrow_length * test_X[2] , color=(.2,.2,.2), arrow_length_ratio=0.1)

    for k in range(len(xs_test)):
        plot_lines(xs_test[k], ys_test[k], zs_test[k])

    # ----- plot the curve and quivers ----- 
    Ws, outputs, losses = get_trace()
    print (outputs.numpy().shape)
    np.save("outputs.npy", outputs.numpy())
    quit()
    
    ax.plot(outputs[:,0], outputs[:,1], outputs[:,2], color = "#dd5533")


    plt.tight_layout()

    plt.legend()
    plt.show()
