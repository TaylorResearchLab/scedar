import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("agg", warn=False)
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.gridspec

import seaborn as sns
sns.set(style="ticks")

import networkx as nx

from . import mtype


def labs_to_cmap(labels, return_lut=False):
    mtype.check_is_valid_labs(labels)

    labels = np.array(labels)
    uniq_lab_arr = np.unique(labels)
    num_uniq_labs = len(uniq_lab_arr)

    lab_col_list = sns.hls_palette(num_uniq_labs)
    lab_cmap = mpl.colors.ListedColormap(lab_col_list)

    if return_lut:
        uniq_lab_lut = dict(zip(range(num_uniq_labs), uniq_lab_arr))
        uniq_ind_lut = dict(zip(uniq_lab_arr, range(num_uniq_labs)))

        lab_ind_arr = np.array([uniq_ind_lut[x] for x in labels])

        lab_col_lut = dict(zip([uniq_lab_lut[i]
                                for i in range(len(uniq_lab_arr))],
                               lab_col_list))
        return (lab_cmap, lab_ind_arr, lab_col_lut, uniq_lab_lut)
    else:
        return lab_cmap


def cluster_scatter(projection2d, labels=None, gradient=None, 
                    title=None, xlab=None, ylab=None,
                    figsize=(20, 20), add_legend=True, n_txt_per_cluster=3,
                    alpha=1, s=0.5, random_state=None, **kwargs):
    projection2d = np.array(projection2d, dtype="float")

    if (projection2d.ndim != 2) or (projection2d.shape[1] != 2):
        raise ValueError("projection2d matrix should have shape "
                         "(n_samples, 2). {}".format(projection2d))

    # TODO: optimize the if-else statement
    if labels is not None:
        mtype.check_is_valid_labs(labels)
        labels = np.array(labels)
        if labels.shape[0] != projection2d.shape[0]:
            raise ValueError(
                "nrow(projection2d matrix) should be equal to len(labels)")
        uniq_labels = np.unique(labels)

        if gradient is not None:
            cmap = kwargs.pop("cmap", "viridis")
            plt.figure(figsize=figsize)
            # matplotlib.collections.PathCollection
            mpc = plt.scatter(x=projection2d[:, 0], y=projection2d[:, 1], 
                              c=gradient, cmap=cmap, s=s, alpha=alpha, 
                              **kwargs)
            if add_legend:
                plt.colorbar(mpc)
            fig = mpc.get_figure()
            ax = fig.get_axes()[0]
        else:
            fig, ax = plt.subplots(figsize=figsize)
            color_lut = dict(zip(uniq_labels,
                                 sns.color_palette("hls", len(uniq_labels))))
            ax.scatter(x=projection2d[:, 0], y=projection2d[:, 1],
                       c=[color_lut[lab] for lab in labels],
                       s=s, alpha=alpha, **kwargs)
            # Add legend
            # Shrink current axis by 20%
            if add_legend:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(handles=[mpl.patches.Patch(color=color_lut[ulab], 
                                                     label=ulab)
                                   for ulab in uniq_labels],
                          bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # randomly select labels for annotation
        if random_state is not None:
            np.random.seed(random_state)

        anno_ind = np.concatenate(
            [np.random.choice(np.where(labels == ulab)[0], n_txt_per_cluster)
             for ulab in uniq_labels])

        for i in map(int, anno_ind):
            ax.annotate(labels[i], (projection2d[i, 0], projection2d[i, 1]))
    else:
        if gradient is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(x=projection2d[:, 0], y=projection2d[:, 1], s=s,
                       alpha=alpha, **kwargs)
        else:
            cmap = kwargs.pop("cmap", "viridis")
            plt.figure(figsize=figsize)
            # matplotlib.collections.PathCollection
            mpc = plt.scatter(x=projection2d[:, 0], y=projection2d[:, 1], 
                              c=gradient, cmap=cmap, s=s, alpha=alpha, 
                              **kwargs)
            if add_legend:
                plt.colorbar(mpc)
            fig = mpc.get_figure()
            ax = fig.get_axes()[0]

    if title is not None:
        ax.set_title(title)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_ylabel(ylab)
    plt.close()
    return fig


def regression_scatter(x, y, title=None, xlab=None, ylab=None, 
                       figsize=(5, 5), alpha=1, s=0.5, ax=None, **kwargs):
    """
    Paired vector scatter plot.
    """
    if xlab is not None:
        x = pd.Series(x, name=xlab)

    if ylab is not None:
        y = pd.Series(y, name=ylab)

    # initialize a new figure
    if ax is None:
        _, ax = plt.subplots()

    ax = sns.regplot(x=x, y=y, ax=ax, **kwargs)

    fig = ax.get_figure()

    if title is not None:
        ax.set_title(title)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_ylabel(ylab)

    fig.set_size_inches(*figsize)
    plt.close()
    return fig

def hist_dens_plot(x, title=None, xlab=None, ylab=None, figsize=(5, 5), 
                   ax=None, **kwargs):
    """
    Plot histogram and density plot of x.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax = sns.distplot(x, norm_hist=None, ax=ax, **kwargs)

    fig = ax.get_figure()

    if title is not None:
        ax.set_title(title)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_ylabel(ylab)

    fig.set_size_inches(*figsize)
    plt.close()
    return fig


def heatmap(x, row_labels=None, col_labels=None, title=None, xlab=None,
            ylab=None, figsize=(20, 20), **kwargs):
    x = np.array(x, dtype="float")
    if x.ndim != 2:
        raise ValueError("x should be 2D array. {}".format(x))

    if x.size == 0:
        raise ValueError("x cannot be empty.")

    if row_labels is not None:
        mtype.check_is_valid_labs(row_labels)
        if len(row_labels) != x.shape[0]:
            raise ValueError("length of row_labels should be the same as the "
                             "number of rows in x."
                             " row_labels: {}. x: {}".format(len(row_labels),
                                                             x.shape))

    if col_labels is not None:
        mtype.check_is_valid_labs(col_labels)
        if len(col_labels) != x.shape[1]:
            raise ValueError("length of col_labels should be the same as the "
                             "number of rows in x."
                             " col_labels: {}. x: {}".format(len(col_labels),
                                                             x.shape))

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    fig = plt.figure(figsize=figsize)
    if title is not None:
        fig.suptitle(title)

    # outer 2x2 grid
    gs = mpl.gridspec.GridSpec(2, 2,
                               width_ratios=[1, 4],
                               height_ratios=[1, 4],
                               wspace=0.0, hspace=0.0)

    # inner upper right for color labels and legends
    ur_gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 height_ratios=[3, 1],
                                                 subplot_spec=gs[1],
                                                 wspace=0.0, hspace=0.0)

    # inner lower left for color labels and legends
    ll_gs = mpl.gridspec.GridSpecFromSubplotSpec(1, 2,
                                                 width_ratios=[3, 1],
                                                 subplot_spec=gs[2],
                                                 wspace=0.0, hspace=0.0)

    ax_lut = {
        'cb_ax': plt.subplot(gs[0]),
        'hm_ax': plt.subplot(gs[3]),
        'lcol_ax': plt.subplot(ll_gs[1]),
        'ucol_ax': plt.subplot(ur_gs[1]),
        'llgd_ax': plt.subplot(ll_gs[0]),
        'ulgd_ax': plt.subplot(ur_gs[0])
    }

    # remove frames and ticks
    for iax in ax_lut.values():
        iax.set_xticks([])
        iax.set_yticks([])
        iax.axis('off')

    # lower right heatmap
    imgp = ax_lut['hm_ax'].imshow(x, cmap='magma', aspect='auto', **kwargs)
    if xlab is not None:
        ax_lut['hm_ax'].set_xlabel(xlab)

    if ylab is not None:
        ax_lut['hm_ax'].set_ylabel(ylab)

    # upper left colorbar
    cb = plt.colorbar(imgp, cax=ax_lut['cb_ax'])
    ax_lut['cb_ax'].set_aspect(5, anchor='W')
    ax_lut['cb_ax'].yaxis.tick_left()
    ax_lut['cb_ax'].axis('on')

    # color labels and legends
    ax_lut['ucol_ax'].set_anchor('S')
    ax_lut['lcol_ax'].set_anchor('E')
    col_axs = (ax_lut['ucol_ax'], ax_lut['lcol_ax'])
    lgd_axs = (ax_lut['ulgd_ax'], ax_lut['llgd_ax'])
    cr_labs = (col_labels, row_labels)
    for i in range(2):
        if cr_labs[i] is not None:
            cmap, ind, ulab_col_lut, ulab_lut = labs_to_cmap(cr_labs[i], 
                                                             return_lut=True)
            if i == 0:
                # col color labels
                ind_mat = ind.reshape(1, -1)
            else:
                # row color labels
                ind_mat = ind.reshape(-1, 1)
            col_axs[i].imshow(ind_mat, cmap=cmap, aspect='auto', **kwargs)

            lgd_patches = [mpl.patches.Patch(color=ulab_col_lut[ulab],
                                             label=ulab)
                           for ulab in sorted(ulab_lut.values())]

            if i == 0:
                # col color legend
                lgd_axs[i].legend(handles=lgd_patches, loc="center", ncol=6)
            else:
                # row color legend
                lgd_axs[i].legend(handles=lgd_patches, loc="upper center",
                                  ncol=1)
    plt.close()
    return fig

def networkx_graph(ng, pos, figsize=(20, 20), node_size=30, alpha=0.05, 
                   with_labels=False, node_color="b", **kwargs):
    fig = plt.figure(figsize=figsize)
    nx.draw_networkx(ng, pos, alpha=alpha, node_color=node_color,
                     node_size=node_size, with_labels=with_labels, **kwargs)
    plt.close()
    return fig

