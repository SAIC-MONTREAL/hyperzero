import math
import random

import numpy
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import seaborn as sns

plt.style.use('seaborn-white')
color_pallete = sns.color_palette('tab20b')


def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def save_fig(name):
    plt.savefig(f"{name}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{name}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()


def create_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def plot_vertices(vertices, plot_type='scatter', ax=None, fig=None,
                  colors=None, bar=False, bar_label="Values", pad=True):
    """
    Plot states and their values.
    """
    if colors is None:
        colors = ["black" for i in range(vertices.shape[0])]
        bar = False
        vmin= -1
        vmax= 1
    else:
        vmin = min(colors)
        vmax = max(colors)

    if plot_type == 'scatter':
        im = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=colors,
                        s=20, zorder=10, edgecolors="black", linewidth=0.5,
                        vmin=vmin, vmax=vmax, alpha=0.3, cmap="cool")
        if bar:
            if pad:
                cbar = fig.colorbar(im, shrink=0.5, pad=0.1)
            else:
                cbar = fig.colorbar(im, shrink=0.5, pad=0)
            im.set_clim(vmin=vmin, vmax=vmax)
            cbar.ax.get_yaxis().labelpad = 10
            cbar.ax.set_ylabel(bar_label, rotation=270)

    elif plot_type == 'plot':
        im = ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                     color=colors, zorder=10, linewidth=1, alpha=0.5)
        # scatter plot start and end points
        im = ax.scatter(vertices[0, 0], vertices[0, 1], vertices[0, 2],
                        color=colors, s=10, zorder=10, linewidth=0.1, alpha=0.5)
        im = ax.scatter(vertices[-1, 0], vertices[-1, 1], vertices[-1, 2],
                        color=colors, s=10, zorder=10, linewidth=0.1, alpha=0.5)

    return ax, fig


def visualize_phase_space(qpos, qvel, values, save_dir, fname, plot_type,
                          goal_coord=None, label='V'):
    # qpos = np.abs(qpos)
    qpos_dim = qpos.shape[-1]
    qvel_dim = qvel.shape[-1]

    is_pendulum = qpos_dim == 1

    if is_pendulum:
        vertices = np.concatenate([qpos, qvel, values], axis=-1)
    else:
        vertices = np.concatenate([qpos, qvel], axis=-1)
    # reshape vertices
    vertices = vertices.reshape(-1, qpos_dim + qvel_dim)
    values = values.reshape(-1, 1)

    # dimensionality reduction if necessary
    if vertices.shape[-1] > 3:
        pca = PCA(n_components=3)
        vertices = pca.fit_transform(vertices)

    vis_colors = random.choice(color_pallete) if plot_type == 'plot' else values

    fig, ax = create_3d_plot()
    ax, fig = plot_vertices(vertices, plot_type=plot_type, fig=fig, ax=ax,
                            colors=vis_colors, bar=False, bar_label='', pad=is_pendulum)

    if is_pendulum:
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\omega$")
        ax.set_zlabel(fr"${label}(\theta, \omega)$")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    if goal_coord is not None:
        assert goal_coord.ndim == 2
        if goal_coord.shape[-1] > 3:
            goal_coord = pca.transform(goal_coord)

        im = ax.scatter(goal_coord[:, 0], goal_coord[:, 1], goal_coord[:, 2], color='green',
                        marker='D', s=100, zorder=1, edgecolors="black", linewidth=0.5, alpha=0.6)

    plt.title(f"{label} on the Phase Space")
    plt.savefig(f"{save_dir}/{fname}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close('all')
