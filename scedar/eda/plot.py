import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.gridspec

import seaborn as sns

import networkx as nx

from . import mtype


mpl.use("agg", warn=False)
sns.set(style="ticks")


def labs_to_cmap(labels, return_lut=False):
    # Each label has its own index and color
    mtype.check_is_valid_labs(labels)

    labels = np.array(labels)
    uniq_lab_arr = np.unique(labels)
    num_uniq_labs = len(uniq_lab_arr)
    uniq_lab_inds = list(range(num_uniq_labs))

    lab_col_list = list(sns.hls_palette(num_uniq_labs))
    lab_cmap = mpl.colors.ListedColormap(lab_col_list)
    # Need to keep track the order of unique labels, so that a labeled
    # legend can be generated.
    # Map unique label indices to unique labels
    uniq_lab_lut = dict(zip(range(num_uniq_labs), uniq_lab_arr))
    # Map unique labels to indices
    uniq_ind_lut = dict(zip(uniq_lab_arr, range(num_uniq_labs)))
    # a list of label indices
    lab_ind_arr = np.array([uniq_ind_lut[x] for x in labels])

    # map unique labels to colors
    # Used to generate legends
    lab_col_lut = dict(zip([uniq_lab_lut[i]
                            for i in range(len(uniq_lab_arr))],
                           lab_col_list))
    # norm separates cmap to difference indices
    # https://matplotlib.org/tutorials/colors/colorbar_only.html
    lab_norm = mpl.colors.BoundaryNorm(uniq_lab_inds + [lab_cmap.N],
                                       lab_cmap.N)
    if return_lut:
        return lab_cmap, lab_norm, lab_ind_arr, lab_col_lut, uniq_lab_lut
    else:
        return lab_cmap, lab_norm


def cluster_scatter(projection2d, labels=None,
                    selected_labels=None,
                    shuffle_label_colors=False, gradient=None,
                    title=None, xlab=None, ylab=None,
                    figsize=(20, 20), add_legend=True, n_txt_per_cluster=3,
                    alpha=1, s=0.5, random_state=None, **kwargs):
    kwargs = kwargs.copy()
    # randomly:
    # - select labels for annotation if required
    # - shuffle colors if required
    np.random.seed(random_state)
    # check projection2d
    projection2d = np.array(projection2d, dtype="float")
    if (projection2d.ndim != 2) or (projection2d.shape[1] != 2):
        raise ValueError("projection2d matrix should have shape "
                         "(n_samples, 2). {}".format(projection2d))
    # check gradient length
    if gradient is not None:
        gradient = np.array(gradient)
        if gradient.ndim != 1:
            raise ValueError("gradient must be 1d.")
        if gradient.shape[0] != projection2d.shape[0]:
            raise ValueError("gradient should have the same length ({}) as "
                             "n_samples in projection2d "
                             "(shape {})".format(gradient.shape[0],
                                                 projection2d.shape[0]))
    # check label length
    if labels is not None:
        mtype.check_is_valid_labs(labels)
        labels = np.array(labels)
        if labels.shape[0] != projection2d.shape[0]:
            raise ValueError("labels should have the same length ({}) as "
                             "n_samples in projection2d "
                             "(shape {})".format(labels.shape[0],
                                                 projection2d.shape[0]))
    # plot selected labels
    if selected_labels is not None:
        if labels is None:
            raise ValueError("selected_labels needs labels to be "
                             "provided.")
        else:
            uniq_selected_labels = np.unique(selected_labels).tolist()
            uniq_labels = np.unique(labels).tolist()
            # np.in1d(uniq_selected_labels, uniq_labels) will cause
            # future warning:
            # https://stackoverflow.com/a/46721064/4638182
            if not np.all([x in uniq_labels
                           for x in uniq_selected_labels]):
                raise ValueError("selected_labels: {} must all "
                                 "be included in the labels: "
                                 "{}.".format(uniq_selected_labels,
                                              uniq_labels))
            slabels_bool = [lab in uniq_selected_labels
                            for lab in labels.tolist()]
            labels = labels[slabels_bool]
            projection2d = projection2d[slabels_bool]
            if gradient is not None:
                gradient = gradient[slabels_bool]

    # TODO: optimize the if-else statement
    if labels is not None:
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
            label_color_arr = np.array(sns.color_palette("hls",
                                                         len(uniq_labels)))
            if shuffle_label_colors:
                np.random.shuffle(label_color_arr)
            color_lut = dict(zip(uniq_labels, label_color_arr))
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
        # [[label1 anno inds], [label2 anno ind], ...]
        anno_ind_list = [np.random.choice(np.where(labels == ulab)[0],
                                          n_txt_per_cluster)
                         for ulab in uniq_labels]
        for ulab_anno in anno_ind_list:
            for i in map(int, ulab_anno):
                ax.annotate(labels[i],
                            (projection2d[i, 0], projection2d[i, 1]))
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


def swarm(x, labels=None, selected_labels=None,
          title=None, xlab=None, ylab=None, figsize=(10, 10), ax=None,
          **kwargs):
    # check x
    x = np.array(x, dtype="float")
    if x.ndim != 1:
        raise ValueError("x should be 1d and have shape "
                         "(n_samples,). {}".format(x))
    if x.shape[0] == 0:
        raise ValueError("x must be non-empty.")
    # check label length
    if labels is not None:
        mtype.check_is_valid_labs(labels)
        labels = np.array(labels)
        if labels.shape[0] != x.shape[0]:
            raise ValueError("labels should have the same length ({}) as "
                             "n_samples in projection2d "
                             "(shape {})".format(labels.shape[0],
                                                 x.shape[0]))
    else:
        # plot selected labels
        if selected_labels is not None:
            raise ValueError("selected_labels needs labels to be "
                             "provided.")
        labels = np.repeat(0, x.shape[0])
    # plot selected labels
    if selected_labels is not None:
        # labels can only be existing
        uniq_selected_labels = np.unique(selected_labels).tolist()
        uniq_labels = np.unique(labels).tolist()
        # np.in1d(uniq_selected_labels, uniq_labels) will cause
        # future warning:
        # https://stackoverflow.com/a/46721064/4638182
        if not np.all([x in uniq_labels
                       for x in uniq_selected_labels]):
            raise ValueError("selected_labels: {} must all "
                             "be included in the labels: "
                             "{}.".format(uniq_selected_labels,
                                          uniq_labels))
        slabels_bool = [lab in uniq_selected_labels
                        for lab in labels.tolist()]
        labels = labels[slabels_bool]
        x = x[slabels_bool]
        if len(x) == 0:
            raise ValueError("No value selected.")

    plt_df = pd.DataFrame({"labels": labels, "val": x})
    if ax is None:
        _, ax = plt.subplots()

    ax = sns.swarmplot(x="labels", y="val", data=plt_df, ax=ax, **kwargs)

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


def heatmap(x, row_labels=None, col_labels=None,
            title=None, xlab=None, ylab=None, figsize=(20, 20),
            transform=None, **kwargs):
    x = np.array(x, dtype="float")
    if x.ndim != 2:
        raise ValueError("x should be 2D array. {}".format(x))

    if x.size == 0:
        raise ValueError("x cannot be empty.")

    if transform is not None:
        if callable(transform):
            # now x must be float, so copy and transform will not cause side
            # effects on original array.
            x = x.copy()
            x = transform(x)
        else:
            raise ValueError("transform must be callable. It will be "
                             "on x.")

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

    im_cmap = kwargs.pop("cmap", "magma")

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
        "cb_ax": plt.subplot(gs[0]),
        "hm_ax": plt.subplot(gs[3]),
        "lcol_ax": plt.subplot(ll_gs[1]),
        "ucol_ax": plt.subplot(ur_gs[1]),
        "llgd_ax": plt.subplot(ll_gs[0]),
        "ulgd_ax": plt.subplot(ur_gs[0])
    }

    # remove frames and ticks
    for iax in ax_lut.values():
        iax.set_xticks([])
        iax.set_yticks([])
        iax.axis("off")

    # lower right heatmap
    imgp = ax_lut["hm_ax"].imshow(x, cmap=im_cmap, aspect="auto", **kwargs)
    if xlab is not None:
        ax_lut["hm_ax"].set_xlabel(xlab)

    if ylab is not None:
        ax_lut["hm_ax"].set_ylabel(ylab)

    # upper left colorbar
    cb = plt.colorbar(imgp, cax=ax_lut["cb_ax"])
    ax_lut["cb_ax"].set_aspect(5, anchor="W")
    ax_lut["cb_ax"].yaxis.tick_left()
    ax_lut["cb_ax"].axis("on")

    # color labels and legends
    ax_lut["ucol_ax"].set_anchor("S")
    ax_lut["lcol_ax"].set_anchor("E")
    col_axs = (ax_lut["ucol_ax"], ax_lut["lcol_ax"])
    lgd_axs = (ax_lut["ulgd_ax"], ax_lut["llgd_ax"])
    cr_labs = (col_labels, row_labels)
    for i in range(2):
        if cr_labs[i] is not None:
            cmap, norm, lab_inds, ulab_col_lut, ulab_lut = labs_to_cmap(
                cr_labs[i], return_lut=True)
            if i == 0:
                # col color labels
                ind_mat = lab_inds.reshape(1, -1)
            else:
                # row color labels
                ind_mat = lab_inds.reshape(-1, 1)
            col_axs[i].imshow(ind_mat, cmap=cmap, norm=norm,
                              aspect="auto", interpolation="nearest")

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


def networkx_graph(ng, pos=None, figsize=(20, 20), node_size=30, alpha=0.05,
                   with_labels=False, node_color="b", **kwargs):
    # TODO: offset labels
    fig = plt.figure(figsize=figsize)
    nx.draw_networkx(ng, pos, alpha=alpha, node_color=node_color,
                     node_size=node_size, with_labels=with_labels, **kwargs)
    plt.close()
    return fig
