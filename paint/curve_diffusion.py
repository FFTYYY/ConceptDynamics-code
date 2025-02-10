import torch
import numpy as np
import os
import glob
import tqdm
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as sim
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

plt.rcParams.update({"font.size": 26})
plot_datass = torch.load("./paint/diffusion_data/plot_datass.pth", weights_only=False)
fig_dir = "./figure/diffusion/"
os.makedirs(fig_dir, exist_ok=True)
sep_cols = []
for key in plot_datass.keys():

    sep_col=key[0]
    sep_cols.append(float(sep_col))
sep_cols=np.unique(np.array(sep_cols))
sep_cols.sort()
plot_datas=plot_datass[list(plot_datass.keys())[0]]
save_steps=np.array(plot_datas[0]["save_steps"])

#cmap="RdYlBu_r"
#cmap = "cool"
norm = mcolors.Normalize(vmin=0.17, vmax=0.7)
colors = plt.get_cmap('Spectral_r')(np.linspace(0, 1, 256))
new_colors = np.vstack((colors[:100], colors[150:]))
cmap = LinearSegmentedColormap.from_list('no_yellow_spectral', new_colors)

def sep_col_to_color(sep_col,cmap):
    #return matplotlib.colormaps[cmap](norm(float(sep_col)))
    return cmap(norm(float(sep_col)))


sel=["0.137", "0.247", "0.358", "0.468", "0.579", "0.689", "0.8  "]
lw=4
fig = plt.figure(figsize=(8, 8))
plt.axline([0,0.5], [1,0.5], c="grey", alpha=0.4, linewidth=0.5)
plt.axline([0.5,0], [0.5,1], c="grey", alpha=0.4, linewidth=0.5)

i_step_min=7
print("min step",save_steps[i_step_min])

sigma_smooth=2.0
plot_mean=True
plot_singles=False

tr_accs=[]
trajs_trss=[]
te_accs=[]
trajs_tess=[]
sep_cols=[]
sep_sizes=[]
for i, key in enumerate(plot_datass.keys()):
    sep_col=key[0]
    sep_size=key[1]
    if sep_col not in sel:
        continue
    sep_cols.append(float(sep_col))
    sep_sizes.append(float(sep_size))
    plot_datas=plot_datass[key]
    classprobs_pred_trs=[]
    classprobs_pred_tes=[]
    seeds=[]
    for plot_data in plot_datas:
        classprobs_pred_tr=np.array(plot_data['classprobs_pred_tr'])[...,1]
        l_tr=np.array(plot_data['l_tr'])
        classprobs_pred_trs.append(classprobs_pred_tr)
        classprobs_pred_te=np.array(plot_data['classprobs_pred_te'])[...,1]
        classprobs_pred_tes.append(classprobs_pred_te)
        l_te=np.array(plot_data['l_te'])
        seeds.append(plot_data["seed"])
    classprobs_pred_trs=np.stack(classprobs_pred_trs,axis=0)
    classprobs_pred_tes=np.stack(classprobs_pred_tes,axis=0)
    #loop for tr
    trajs_trs=[]
    tr_accs_=[]
    for i in range(3):
        trajs_trs.append(classprobs_pred_trs[:,:,:,l_tr==i].mean((0,3)))
        gt=np.array([[0,0],[0,1],[1,0]][i])
        tr_accs_.append(1-np.abs(gt[None,None,:,None]-classprobs_pred_trs[:,:,:,l_tr==i]).mean(3))
    trajs_trs=np.stack(trajs_trs,axis=0)
    tr_accs.append(np.stack(tr_accs_,axis=0))
    #in 2D only 1 test
    te_accs.append(classprobs_pred_tes[:,:,:,l_te==3].mean(3))
    trajs_te=classprobs_pred_tes[:,:,:,l_te==3].mean((0,3))

    trajs_trs=sim.gaussian_filter1d(trajs_trs,sigma_smooth,axis=1,truncate=6.0)
    trajs_te=sim.gaussian_filter1d(trajs_te,sigma_smooth,axis=0,truncate=6.0)
    trajs_trss.append(trajs_trs)
    trajs_tess.append(trajs_te)
    color=sep_col_to_color(sep_col,cmap=cmap)
    #color=cfg_to_color(cfg,cmap=cmap)

    if plot_mean:
        plt.plot(trajs_te[i_step_min:,0],trajs_te[i_step_min:,1],c=color,markersize=5,linewidth=5,alpha=0.6)
        for pos in range(i_step_min, len(trajs_te) - i_step_min, 6):
            plt.annotate(
             '', xy=(trajs_te[pos + 1, 0], trajs_te[pos + 1, 1]), xytext=(trajs_te[pos,0], trajs_te[pos,1]),
             arrowprops=dict(arrowstyle='-|>', color=color, lw=0, alpha=0.6, mutation_scale=30)
            )
    if plot_singles:
        for i_seed,traj_te_single in enumerate(classprobs_pred_tes[:,:,:,l_te==3].mean(3)):
            seed=seeds[i_seed]
            traj_smooth=sim.gaussian_filter1d(traj_te_single,sigma_smooth,axis=0,truncate=6.0)
            plt.plot(traj_smooth[i_step_min:,0],traj_smooth[i_step_min:,1],c=color,alpha=0.3)


tr_accs=np.stack(tr_accs,axis=0)
te_accs=np.stack(te_accs,axis=0)
trajs_trss=np.stack(trajs_trss,axis=0)
trajs_tess=np.stack(trajs_tess,axis=0)
sep_cols=np.array(sep_cols)
#plt.ylabel("Size",labelpad=-60)
#plt.xlabel("Color",labelpad=-60)
plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.xticks([])
plt.yticks([])
ax = plt.gca()
ax.spines["top"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)
ax.spines["right"].set_linewidth(0.5)
ax.spines["left"].set_linewidth(0.5)
ax.tick_params(axis="both", which="both", bottom=True, top=False,
               labelbottom=True, left=True, right=False,
               labelleft=True,direction='out',length=5,width=1.0,pad=8)
#plt.legend(loc="upper center",ncol=2)

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cfg)
cbar = plt.colorbar(sm,cax=cax,fraction=0.046, pad=0.04)#adjust size
cbar.set_label("Color Concept Signal")
ax.set_xlabel("Color",fontsize=35)
ax.set_ylabel("Size",fontsize=35)
circle1 = Circle((0.87, 0.07), 0.02, transform=fig.transFigure, color='#00A0FF', fill=True)
circle2 = Circle((0.10, 0.07), 0.02, transform=fig.transFigure, color='#F0200A', fill=True)
circle3 = Circle((0.10, 0.925), 0.012, transform=fig.transFigure, color='#F0200A', fill=True)
circle4 = Circle((0.87, 0.925), 0.012, transform=fig.transFigure, color='#00A0FF', fill=True)
fig.add_artist(circle1)
fig.add_artist(circle2)
fig.add_artist(circle3)
fig.add_artist(circle4)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(os.path.join(fig_dir,"concept_signal.pdf"),bbox_inches="tight")



colors = {0: "#56C1FF", 1: "#FF95CA", 2: "#ED220D", 3: "#ff42a1"}
dist=((trajs_tess-np.array([1.,1.])[None,None,:])**2).mean(-1)
plt.figure(figsize=(8,6))

indexed_sep_cols = list(enumerate(sep_cols))
sorted_indexed = sorted(indexed_sep_cols, key=lambda x: x[1])
sorted_indices = [index for index, value in sorted_indexed]
for i in sorted_indices:
    ##plt.plot(save_steps,dist[i],label="$\Delta Col=$"+str(sel[i]),linewidth=lw)
    color=sep_col_to_color(sep_cols[i],cmap=cmap)
    plt.plot(save_steps,dist[i],label=str(sep_cols[i]),linewidth=3.5, color=color, alpha=0.8)
plt.legend(title="Color Concept Signal", ncol=3,bbox_to_anchor=(0.5, 1.21), loc='center', fontsize=25, columnspacing=1)
plt.xlabel("Number of gradient steps $t$")
plt.ylabel("Concept Space MSE")
plt.xscale("log")
ax = plt.gca()
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["left"].set_linewidth(2)
ax.tick_params(axis="both", which="both", bottom=True, top=False,
               labelbottom=True, left=True, right=False,
               labelleft=True,direction='out',length=5,width=1.0,pad=8,labelsize=26)
plt.savefig(os.path.join(fig_dir,"double_descent_concept_mse.pdf"),bbox_inches="tight")

import seaborn as sns
te_comb=te_accs.prod(-1).mean(1)
sigma_smooth=1.0
te_comb_smooth=sim.gaussian_filter1d(te_comb,sigma_smooth,axis=1,truncate=6.0)
plt.figure(figsize=(8,6))
for i in sorted_indices:
    color=sep_col_to_color(sep_cols[i],cmap=cmap)
    plt.plot(save_steps,te_comb_smooth[i],label=str(sep_cols[i]),linewidth=3.5, color=color, alpha=0.8)
plt.legend(title="Color Concept Signal", ncol=3,bbox_to_anchor=(0.5, 1.21), loc='center', fontsize=25, columnspacing=1)
plt.ylim(0,1)
plt.xscale("log")
ax = plt.gca()
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["left"].set_linewidth(2)
ax.tick_params(axis="both", which="both", bottom=True, top=False,
               labelbottom=True, left=True, right=False,
               labelleft=True,direction='out',length=5,width=1.0,pad=8,labelsize=26)
plt.savefig(os.path.join(fig_dir,"double_descent_combined_multiplicative_acc.pdf"),bbox_inches="tight")

m_te_accs=te_accs.mean(1)
grad=np.gradient(m_te_accs,save_steps,axis=1)
speed=np.linalg.norm(grad,axis=-1)
sigma_smooth=1.5
speed_smooth=sim.gaussian_filter1d(speed,sigma_smooth,axis=1,truncate=6.0)
plt.figure(figsize=(8,6))
for i in sorted_indices:
    color=sep_col_to_color(sep_cols[i],cmap=cmap)
    plt.plot(save_steps,speed_smooth[i],label=str(sep_cols[i]),linewidth=3.5, color=color, alpha=0.8)
#plt.xlabel("Steps [t]")
#plt.ylabel("$|dC/dt|$")
plt.yscale("log")
#plt.xscale("log")
plt.legend(title="Color Concept Signal", ncol=3,bbox_to_anchor=(0.5, 1.2), loc='center', fontsize=25, columnspacing=1)
plt.ylim(bottom=1e-5)
ax = plt.gca()
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["left"].set_linewidth(2)
ax.tick_params(axis="both", which="both", bottom=True, top=False,
               labelbottom=True, left=True, right=False,
               labelleft=True,direction='out',length=5,width=1.0,pad=8,labelsize=26)
ax.set_ylabel("$|\mathrm{d}C/\mathrm{d}t|$")
ax.set_xlabel("Number of gradient steps $t$")
plt.savefig(os.path.join(fig_dir,"concept_learning_speed.pdf"),bbox_inches="tight")
