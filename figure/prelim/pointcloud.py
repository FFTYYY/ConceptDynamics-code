import torch as tc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d

def plot_lines(point_x, point_y, point_z):
    ax.plot([point_x, point_x], [point_y, 0], [point_z, point_z], linestyle="dashed", color="gray")
    ax.plot([point_x, 0], [point_y, point_y], [point_z, point_z], linestyle="dashed", color="gray")
    ax.plot([point_x, point_x], [point_y, point_y], [point_z, 0], linestyle="dashed", color="gray")

def copy(x: list, k: int):
    return x if k <= 1 else copy(x + x, k // 2)

if __name__ == "__main__":
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = tc.stack( [ tc.randn(1000) * 0.05 + 1, tc.randn(1000) * 0.05, tc.randn(1000) * 0.05], dim = 0).view(-1)
    ys = tc.stack( [ tc.randn(1000) * 0.05, tc.randn(1000) * 0.05 + 1, tc.randn(1000) * 0.05], dim = 0).view(-1)
    zs = tc.stack( [ tc.randn(1000) * 0.05, tc.randn(1000) * 0.05, tc.randn(1000) * 0.05 + 1], dim = 0).view(-1)

    ax.scatter(xs, ys, zs, label = "train set")

    xs_test = copy([0, 1, 1, 1, ]  , 8)
    ys_test = copy([1, 1, 0, 1, ]  , 8)
    zs_test = copy([1, 0, 1, 1, ]  , 8)
    ax.scatter(xs_test, ys_test, zs_test, marker = "^", s = 200, label = "test points")

    arrow_length = 1.3
    ax.quiver(0, 0, 0, arrow_length, 0, 0, color=(.2,.2,.2), arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, arrow_length, 0, color=(.2,.2,.2), arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, arrow_length, color=(.2,.2,.2), arrow_length_ratio=0.1)

    for k in range(len(xs_test)):
        plot_lines(xs_test[k], ys_test[k], zs_test[k])

    plt.tight_layout()

    plt.legend()
    plt.show()
