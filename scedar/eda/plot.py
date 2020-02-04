import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("agg", warn=False)  # noqa
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns

import networkx as nx

from scedar.eda import mtype

from collections import OrderedDict

sns.set(style="ticks")


def labs_to_cmap(labels, return_lut=False, shuffle_colors=False,
                 random_state=None):
    np.random.seed(random_state)
    # Each label has its own index and color
    mtype.check_is_valid_labs(labels)

    labels = np.array(labels)
    uniq_lab_arr = np.unique(labels)
    num_uniq_labs = len(uniq_lab_arr)
    uniq_lab_inds = list(range(num_uniq_labs))

    lab_col_list = list(sns.hls_palette(num_uniq_labs))
    if shuffle_colors:
        np.random.shuffle(lab_col_list)

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
                    plot_different_markers=False,
                    label_markers=None,
                    shuffle_label_colors=False, gradient=None,
                    xlim=None, ylim=None,
                    title=None, xlab=None, ylab=None,
                    figsize=(20, 20), add_legend=True, n_txt_per_cluster=3,
                    alpha=1, s=0.5, random_state=None, **kwargs):
    """Scatter plot for clustering illustration

    Args:
        projection2d (2 col numeric array): (n, 2) matrix to plot
        labels (list of labels): labels of n samples
        selected_labels (list of labels): selected labels to plot
        plot_different_markers (bool): plot different markers for samples with
            different labels
        label_markers (list of marker shapes): passed to matplotlib plot
        shuffle_label_colors (bool): shuffle the color of labels to avoid
            similar colors show up in close clusters
        gradient (list of number): color gradient of n samples
        title (str)
        xlab (str): x axis label
        ylab (str): y axis label
        figsize (tuple of two number): (width, height)
        add_legend (bool)
        n_txt_per_cluster (number): the number of text to plot per cluster.
            Could be 0.
        alpha (number)
        s (number): size of the points
        random_state (int): random seed to shuffle features
        **kwards: passed to matplotlib plot

    Return:
        matplotlib figure of the created scatter plot
    """
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
    # check markers
    if label_markers is not None:
        if labels is None:
            raise ValueError("labels should not be None when label_markers")
        if len(label_markers) != len(labels):
            raise ValueError("labels should have the same length as"
                             "label_markers")
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
    fig, ax = plt.subplots(figsize=figsize)
    # TODO: optimize the if-else statement
    if labels is not None:
        # return empty scatter plot if there is no point to plot
        # markers for each label
        uniq_labels = np.unique(labels)
        # create marker dict:
        # lab_m_s_ind_lut: {(lab1, marker1): marker_1_s_ind_list}
        lab_m_s_ind_lut = {}
        if plot_different_markers:
            if label_markers is None:
                # use a different marker for each label
                # cycle use the following filled markers:
                # "o": circle
                # "s": square
                # "^": triangle_up
                # "D": diamond
                # "x": x
                # "v": triangle_down
                # "d": thin_diamond
                # "+": plus
                # ">": triangle_right
                # "p": pentagon
                # "h": hexagon1
                # "<": triangle_left
                # "H": hexagon2
                # "*": star
                # order: "os^Dxvd+>ph<H*"
                # Refs:
                # - markers with shape:
                # https://matplotlib.org/examples/lines_bars_and_markers/
                # marker_reference.html
                # - all merkers
                # ref: https://matplotlib.org/api/markers_api.html
                m_cycle = "os^Dxvd+>ph<H*"
                for i, ulab in enumerate(uniq_labels):
                    ulab_m = m_cycle[i % len(m_cycle)]
                    lab_m_s_ind_lut[(ulab, ulab_m)] = list(filter(
                        lambda j: labels[j] == ulab, range(len(labels))))
            else:
                # use user provided markers
                for ulab_m in set(label_markers):
                    # if user provided, legend show marker rather than
                    # label
                    lab_m_s_ind_lut[(ulab_m, ulab_m)] = list(filter(
                        lambda i: label_markers[i] == ulab_m,
                        range(len(label_markers))))
        else:
            # plot all labels with shape "o"
            for i, ulab in enumerate(uniq_labels):
                lab_m_s_ind_lut[(ulab, "o")] = list(filter(
                    lambda j: labels[j] == ulab, range(len(labels))))
        # plot
        # list of ulabs
        lgd_ulabs = []
        # list of matplotlib.collections.PathCollection
        lgd_mpcs = []
        if gradient is not None:
            cmap = kwargs.pop("cmap", "viridis")
            # lab_m_s_ind_lut = {(lab1, m1): [s_inds]}
            for (ulab, ulab_m), s_inds in sorted(lab_m_s_ind_lut.items()):
                mpc = plt.scatter(x=projection2d[s_inds, 0],
                                  y=projection2d[s_inds, 1],
                                  c=gradient[s_inds], cmap=cmap,
                                  marker=ulab_m,
                                  s=s, alpha=alpha,
                                  **kwargs)
                lgd_ulabs.append(ulab)
                lgd_mpcs.append(mpc)
            if add_legend and len(labels) != 0:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
                plt.legend(handles=lgd_mpcs, labels=lgd_ulabs,
                           bbox_to_anchor=(1.25, 1), loc=2,
                           borderaxespad=0.)
                # colorbar location
                # ref:
                # https://matplotlib.org/gallery/axes_grid1/
                # demo_colorbar_with_inset_locator.html
                cb_axins = inset_axes(
                    ax,
                    width="5%",  # width = 10% of parent_bbox width
                    height="100%",  # height : 50%
                    loc=2,
                    bbox_to_anchor=(1.05, 0., 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
                plt.colorbar(cax=cb_axins)
        else:
            label_color_arr = np.array(
                sns.color_palette("hls", len(uniq_labels)))
            if shuffle_label_colors:
                np.random.shuffle(label_color_arr)
            color_lut = dict(zip(uniq_labels, label_color_arr))
            s_col_arr = np.array([color_lut[lab] for lab in labels])
            # lab_m_s_ind_lut = {(lab1, m1): [s_inds]}
            for (ulab, ulab_m), s_inds in sorted(lab_m_s_ind_lut.items()):
                mpc = plt.scatter(x=projection2d[s_inds, 0],
                                  y=projection2d[s_inds, 1],
                                  c=s_col_arr[s_inds],
                                  marker=ulab_m,
                                  s=s, alpha=alpha,
                                  **kwargs)
                lgd_ulabs.append(ulab)
                lgd_mpcs.append(mpc)
            # Add legend
            # Shrink current axis by 20%
            if add_legend and len(labels) != 0:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
                plt.legend(handles=lgd_mpcs, labels=lgd_ulabs,
                           bbox_to_anchor=(1.05, 1), loc=2,
                           borderaxespad=0.)
        # add text annotation [[label1 anno inds], [label2 anno ind], ...]
        anno_ind_list = [np.random.choice(np.where(labels == ulab)[0],
                                          n_txt_per_cluster)
                         for ulab in uniq_labels]
        for ulab_anno in anno_ind_list:
            for i in map(int, ulab_anno):
                ax.annotate(labels[i],
                            (projection2d[i, 0], projection2d[i, 1]))
    else:
        if gradient is None:
            plt.scatter(x=projection2d[:, 0], y=projection2d[:, 1], s=s,
                        alpha=alpha, **kwargs)
        else:
            cmap = kwargs.pop("cmap", "viridis")
            # matplotlib.collections.PathCollection
            plt.scatter(x=projection2d[:, 0], y=projection2d[:, 1],
                        c=gradient, cmap=cmap, s=s, alpha=alpha,
                        **kwargs)
            if add_legend:
                plt.colorbar()

    if title is not None:
        ax.set_title(title)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_ylabel(ylab)

    plt.xlim(xlim)
    plt.ylim(ylim)
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
            transform=None, shuffle_row_colors=False,
            shuffle_col_colors=False, random_state=None,
            row_label_order=None, col_label_order=None, **kwargs):
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
            if i == 0:
                # col color labels
                cmap, norm, lab_inds, ulab_col_lut, ulab_lut = labs_to_cmap(
                    cr_labs[i], return_lut=True,
                    shuffle_colors=shuffle_col_colors,
                    random_state=random_state)
                ind_mat = lab_inds.reshape(1, -1)
                if col_label_order is None:
                    lgd_patches = [mpl.patches.Patch(color=ulab_col_lut[ulab],
                                                     label=ulab)
                                   for ulab in sorted(ulab_lut.values())]
                else:
                    lgd_patches = [mpl.patches.Patch(color=ulab_col_lut[ulab],
                                                     label=ulab)
                                   for ulab in col_label_order]
            else:
                # row color labels
                cmap, norm, lab_inds, ulab_col_lut, ulab_lut = labs_to_cmap(
                    cr_labs[i], return_lut=True,
                    shuffle_colors=shuffle_row_colors,
                    random_state=random_state)
                ind_mat = lab_inds.reshape(-1, 1)
                if row_label_order is None:
                    lgd_patches = [mpl.patches.Patch(color=ulab_col_lut[ulab],
                                                     label=ulab)
                                   for ulab in sorted(ulab_lut.values())]
                else:
                    lgd_patches = [mpl.patches.Patch(color=ulab_col_lut[ulab],
                                                     label=ulab)
                                   for ulab in row_label_order]

            col_axs[i].imshow(ind_mat, cmap=cmap, norm=norm,
                              aspect="auto", interpolation="nearest")

            if i == 0:
                # col color legend
                lgd_axs[i].legend(handles=lgd_patches, loc="center", ncol=6)
            else:
                # row color legend
                lgd_axs[i].legend(handles=lgd_patches, loc="upper center",
                                  ncol=1)
    plt.close()
    return fig


def networkx_graph(ng, pos=None, alpha=0.05, figsize=(20, 20), gradient=None,
                   labels=None, different_label_markers=True, node_size=30,
                   node_with_labels=False, nx_draw_kwargs=None):
    # TODO: offset labels
    fig = plt.figure(figsize=figsize)

    if nx_draw_kwargs is None:
        nx_draw_kwargs = {}

    if labels is not None:
        # prepare markers and colors for each unique label
        if different_label_markers:
            # each marker for each label
            # use a different marker for each label
            # cycle use the following filled markers:
            # "o": circle
            # "s": square
            # "^": triangle_up
            # "D": diamond
            # "x": x
            # "v": triangle_down
            # "d": thin_diamond
            # "+": plus
            # ">": triangle_right
            # "p": pentagon
            # "h": hexagon1
            # "<": triangle_left
            # "H": hexagon2
            # "*": star
            # order: "os^Dxvd+>ph<H*"
            # Refs:
            # - markers with shape:
            # https://matplotlib.org/examples/lines_bars_and_markers/
            # marker_reference.html
            # - all merkers
            # ref: https://matplotlib.org/api/markers_api.html
            m_cycle = "os^Dxvd+>ph<H*"
        else:
            # all labels use "o"
            m_cycle = "o"
        # each label has a marker
        uniq_labels = sorted(set(labels))
        ulab_colors = sns.hls_palette(len(uniq_labels))
        lab_m_s_ind_lut = OrderedDict()
        for i, ulab in enumerate(uniq_labels):
            ulab_m = m_cycle[i % len(m_cycle)]
            ulab_c = ulab_colors[i]
            lab_m_s_ind_lut[(ulab, ulab_m, ulab_c)] = list(filter(
                lambda j: labels[j] == ulab, range(len(labels))))

    if labels is None:
        if gradient is None:
            # no label. no gradient.
            # plot all nodes as blue.
            node_color = nx_draw_kwargs.pop("node_color", "b")
            cmap = nx_draw_kwargs.pop("cmap", None)
            nx.draw_networkx(ng, pos, alpha=alpha, node_color=node_color,
                             cmap=cmap, node_size=node_size,
                             with_labels=node_with_labels,
                             **nx_draw_kwargs)
        else:
            # no label. has gradient.
            cmap = nx_draw_kwargs.pop("cmap", "viridis")
            # matplotlib.collections.PathCollection
            nx.draw_networkx_edges(ng, pos, alpha=alpha)
            mcp = nx.draw_networkx_nodes(ng, pos, alpha=alpha,
                                         node_color=gradient, cmap=cmap,
                                         node_size=node_size,
                                         with_labels=node_with_labels,
                                         **nx_draw_kwargs)
            plt.colorbar(mcp)
    else:
        nx.draw_networkx_edges(ng, pos, alpha=alpha)
        if gradient is None:
            # has label. no gradient.
            # plot differnt labels with different colors and markers.
            for (ulab, ulab_m, ulab_c), ulab_s_inds in lab_m_s_ind_lut.items():
                mcp = nx.draw_networkx_nodes(ng, pos, alpha=alpha,
                                             nodelist=ulab_s_inds,
                                             node_color=ulab_c,
                                             node_shape=ulab_m,
                                             node_size=node_size, label=ulab)
        else:
            # has label. has gradient.
            gradient = np.array(gradient)
            cmap = nx_draw_kwargs.pop("cmap", "viridis")
            for (ulab, ulab_m, ulab_c), ulab_s_inds in lab_m_s_ind_lut.items():
                mcp = nx.draw_networkx_nodes(ng, pos, alpha=alpha,
                                             nodelist=ulab_s_inds,
                                             node_color=gradient[ulab_s_inds],
                                             node_shape=ulab_m,
                                             node_size=node_size,
                                             label=ulab, cmap=cmap)
            # TODO: move legend out of the graph
            plt.colorbar(mcp)
        # TODO: move legend out of the graph
        plt.legend(scatterpoints=1)
    plt.close(fig)
    return fig
