import itertools
import numpy as np

from scedar import eda
from scedar.eda.slcs import MDLSingleLabelClassifiedSamples as MDLSLCS
from scedar.eda.slcs import SingleLabelClassifiedSamples as SLCS
from scedar import utils


class MIRAC(object):
    """
    MIRAC: MDL iteratively regularized agglomerative clustering.

    Args
    ----
    x : float array
        Data matrix.
    d : float array
        Distance matrix.
    metric : str
        Type of distance metric.
    sids : sid list
        List of sample ids.
    fids : fid list
        List of feature ids.
    hac_tree : HCTree
        Hierarchical tree built by agglomerative clustering
        to divide in MIRAC. If provided, distance matrix will not be used
        for building another tree.
    nprocs : int
        Number of processes to run MIRAC parallely.
    cl_mdl_scale_factor : float
        Scale factor of cluster overhead mdl.
    min_cl_n : int
        Minimum # samples in a cluster.
    encode_type : {"auto", "data", or "distance"}
        Type of values to encode. If "auto", encode data when
        n_features <= 100.
    mdl_method : mdl.Mdl
        If None, use ZeroIGKdeMdl for encoded values
        with >= 50% zeros, and use GKdeMdl otherwise.
    linkage : str
        Linkage type for generating the hierarchy.
    optimal_ordering : bool
        To require hierarchical clustering tree with optimal ordering. Default
        value is False.
    dim_reduct_method : {"PCA", "t-SNE", "UMAP", None}
        If None, no dimensionality reduction before clustering.
    verbose : bool
        Print stats for each iteration.

    Attributes
    ----------
    _sdm : SampleDistanceMatrix
        Data and distance matrices.
    _min_cl_n : int
        Stored parameter.
    _encode_type : str
        Encode type. If "auto" provided, this attribute
        will store the determined encode type.
    _mdl_method : mdl.Mdl
        Mdl method. If None is provided, this attribute
        will store the determined mdl method.
    labs : label list
        Labels of clustered samples. 1-to-1 matching to
        from first to last.
    _hac_tree : eda.hct.HClustTree
        Root node of the hierarchical agglomerative clustering tree.
    _run_log : str
        String containing the log of the MIRAC run.

    TODO:
    * Dendrogram representation of the splitting process.

    * Take HCTree as parameter. Computing it is non-trivial when n is large.

    * Simplify splitting criteria.
    """

    # TODO: use PCA/tsne/umap
    def __init__(self, x, d=None, metric="cosine", sids=None, fids=None,
                 hac_tree=None, nprocs=1, cl_mdl_scale_factor=1,
                 min_cl_n=25, encode_type="auto", mdl_method=None,
                 min_split_mdl_red_ratio=0.2,
                 soft_min_subtree_size=1,
                 linkage="complete", optimal_ordering=False,
                 dim_reduct_method=None,
                 verbose=False):
        super().__init__()
        # initialize simple attributes
        self._nprocs = max(int(nprocs), 1)
        self._is_euc_dist = metric == "euclidean"
        self._verbose = verbose
        self._linkage = linkage
        self._optimal_ordering = optimal_ordering
        self._dim_reduct_method = dim_reduct_method
        # check dimensionality reduction method
        if dim_reduct_method is not None:
            # TODO: use pdist if provided
            dim_red_sdm = eda.SampleDistanceMatrix(
                x=x, metric=metric, use_pdist=False, nprocs=nprocs)
            if dim_reduct_method == "PCA":
                data_x = dim_red_sdm._pca_x
            elif dim_reduct_method == "t-SNE":
                data_x = dim_red_sdm.tsne(
                    n_iter=3000, random_state=17, verbose=verbose)
            elif dim_reduct_method == "UMAP":
                data_x = dim_red_sdm._umap_x
            else:
                raise ValueError("Not supported dimensionality reduction "
                                 "method: {}".format(dim_reduct_method))
        else:
            data_x = x
        # labels for computing MDL
        self._sdm = MDLSLCS(x=data_x, labs=[0]*data_x.shape[0],
                            d=d, metric=metric,
                            sids=sids, fids=fids, encode_type=encode_type,
                            mdl_method=mdl_method, nprocs=nprocs)
        self._encode_type = self._sdm._encode_type
        self._mdl_method = self._sdm._mdl_method
        # initialize hierarchical clustering tree
        if hac_tree is not None:
            n_leaf_nodes = len(hac_tree.leaf_ids())
            if n_leaf_nodes != self._sdm._x.shape[0]:
                raise ValueError("hac_tree should have same number of "
                                 "samples as x.")
        else:
            hac_tree = eda.HClustTree.hclust_tree(
                self._sdm._d, linkage=self._linkage,
                optimal_ordering=self._optimal_ordering,
                is_euc_dist=self._is_euc_dist, verbose=self._verbose)
        self._hac_tree = hac_tree
        # run
        self.tune_parameters(cl_mdl_scale_factor, min_cl_n,
                             min_split_mdl_red_ratio,
                             soft_min_subtree_size, self._verbose)

    def _set_parameters(self, cl_mdl_scale_factor=1, min_cl_n=25,
                        min_split_mdl_red_ratio=0.2,
                        soft_min_subtree_size=1):
        # initialize cluster mdl scale factor
        if cl_mdl_scale_factor < 0:
            raise ValueError("cl_mdl_scale_factor should >= 0"
                             "cl_mdl_scale_factor: "
                             "{}".format(cl_mdl_scale_factor))
        self._cl_mdl_scale_factor = cl_mdl_scale_factor
        # intialize min_cl_n
        # assert min_cl_n > 0
        min_cl_n = int(min_cl_n)
        if min_cl_n <= 0:
            raise ValueError("min_cl_n shoud > 0. "
                             "min_cl_n: {}".format(min_cl_n))
        self._min_cl_n = min_cl_n
        self._min_split_mdl_red_ratio = min_split_mdl_red_ratio
        self._soft_min_subtree_size = soft_min_subtree_size
        return

    def tune_parameters(self, cl_mdl_scale_factor=1, min_cl_n=25,
                        min_split_mdl_red_ratio=0.2,
                        soft_min_subtree_size=1, verbose=False):
        self._verbose = verbose
        self._set_parameters(cl_mdl_scale_factor, min_cl_n,
                             min_split_mdl_red_ratio,
                             soft_min_subtree_size)
        # run MIRAC with initialized parameters
        s_inds, s_labs = self._mirac()
        # initialize labels
        self._labs = s_labs[np.argsort(s_inds, kind="mergesort")].tolist()
        return

    def dmat_heatmap(self, selected_labels=None, col_labels=None,
                     transform=None,
                     title=None, xlab=None, ylab=None, figsize=(10, 10),
                     **kwargs):
        # hierarchical clustering tree leaf sample inds ordered from
        # left to right
        leaf_order = self._hac_tree.leaf_ids()
        if len(leaf_order) == 0:
            return None
        # leaf labels from left to right
        leaf_ordered_labs = np.array(self._labs)[leaf_order].tolist()
        # check labels not interrupted in leaf order
        # in other word, same labels should be adjacent to each other
        # Examples:
        # - good: [1] and [1, 1, 2, 2, 3]
        # - bad: [1, 2, 1, 1, 3, 3]
        curr_lab = leaf_ordered_labs[0]
        lab_set = set([curr_lab])
        for ilab in leaf_ordered_labs:
            if ilab != curr_lab:
                # reached the next group of labels
                if ilab in lab_set:
                    raise ValueError("Same labels should be grouped "
                                     "together.\n\t"
                                     "iterating lab: {}\n\t"
                                     "iterated lab set: {}\n\t"
                                     "leaf order: {}\n\t"
                                     "leaf ordered labs: {}".format(
                                         ilab, lab_set, leaf_order,
                                         leaf_ordered_labs))
                lab_set.add(ilab)
                curr_lab = ilab
        # generate heatmap
        # select labels to plot
        s_lab_bool_inds = SLCS.select_labs_bool_inds(
            leaf_ordered_labs, selected_labels)
        s_leaf_order = list(itertools.compress(leaf_order, s_lab_bool_inds))
        s_leaf_ordered_labs = list(itertools.compress(leaf_ordered_labs,
                                                      s_lab_bool_inds))
        s_d = self._sdm._d[s_leaf_order][:, s_leaf_order]
        return eda.heatmap(s_d, row_labels=s_leaf_ordered_labs,
                           col_labels=col_labels, transform=transform,
                           title=title, xlab=xlab, ylab=ylab,
                           figsize=figsize, **kwargs)

    @property
    def labs(self):
        return self._labs.copy()

    def _mirac(self):
        # iterative hierarchical agglomerative clustering
        # Input:
        # - cl_mdl_scale_factor: scale cluster overhead mdl
        # - minimax_n: estimated minimum # samples in a cluster
        # - maxmini_n: estimated max # samples in a cluster.
        #   If none, 10 * minimax is used.
        leaf_order = self._hac_tree.leaf_ids()

        n_samples = self._sdm._x.shape[0]
        n_features = self._sdm._x.shape[1]

        self._run_log = ""

        # split samples into sub-clusters with less than min_cl_n samples
        curr_trees = [self._hac_tree]
        next_trees = []
        split_s_inds = []
        while len(curr_trees) != 0:
            # Split each of the hac tree in curr_trees
            for i in range(len(curr_trees)):
                # lst, rst: left_sub_tree, right_sub_tree
                labs, s_inds, lst, rst = curr_trees[i].bi_partition(
                    soft_min_subtree_size=self._soft_min_subtree_size,
                    return_subtrees=True)
                s_cnt = len(s_inds)
                subtrees = [lst, rst]
                subtree_s_ind_list = [t.leaf_ids() for t in subtrees]
                subtree_s_cnt_list = [len(x) for x in subtree_s_ind_list]
                n_subtrees = len(subtrees)

                subtree_split_list = []
                for st_ind in range(n_subtrees):
                    if subtree_s_cnt_list[st_ind] < self._min_cl_n:
                        subtree_split_list.append("min-cl-size")
                        split_s_inds.append(subtree_s_ind_list[st_ind])
                    else:
                        subtree_split_list.append("split")
                        next_trees.append(subtrees[st_ind])

                curr_iter_run_log = str.format(
                    "subtree_n: {}, "
                    "subtree_n/n: {},"
                    "split: {}.\n",
                    subtree_s_cnt_list,
                    [x / s_cnt for x in subtree_s_cnt_list],
                    subtree_split_list)
                self._run_log += curr_iter_run_log

            curr_trees = next_trees
            next_trees = []
        # sort individual splitted sample index list
        sub_cl_order = np.argsort(list(map(lambda x: leaf_order.index(x[0]),
                                           split_s_inds)))
        split_s_inds = [split_s_inds[i] for i in sub_cl_order]
        # merge subclusters by mdl
        # start with merging clusters at beginning to form a cluster with more
        # than min_cl_n samples.
        while len(split_s_inds) > 1 and len(split_s_inds[0]) < self._min_cl_n:
            # pop first 2 sub-cluster indices
            # concatenate
            # insert concatenated cluster back at the beginning
            split_s_inds.insert(0, split_s_inds.pop(0) + split_s_inds.pop(0))
        # Then merge clusters at the end to form a cluster with more than
        # min_cl_n samples
        while len(split_s_inds) > 1 and len(split_s_inds[-1]) < self._min_cl_n:
            # pop last 2 sub-cluster indices
            # concatenate
            # append at the end
            split_s_inds.append(split_s_inds.pop(-2) + split_s_inds.pop(-1))
        # merge other sub-clusters
        # m_ind points to the left sub-cluster (scl_left) of the sub-cluster
        # currently being inspected (scl_insp)
        m_ind = 0
        while m_ind < len(split_s_inds) - 1:
            scl_left = split_s_inds.pop(m_ind)
            scl_insp = split_s_inds.pop(m_ind)
            while len(scl_insp) < self._min_cl_n and len(split_s_inds) > m_ind:
                scl_right = split_s_inds.pop(m_ind)
                # rhs sub-cluster with >= min_cl_n samples for mdl encoding
                scl_r_minimax = scl_right.copy()
                # concatenate sub-clasters after scl_r_minimax
                scl_r_enc_i = m_ind
                while (len(scl_r_minimax) < self._min_cl_n and
                       scl_r_enc_i < len(split_s_inds)):
                    # += edits list in-place, so copy is necessary
                    scl_r_minimax += split_s_inds[scl_r_enc_i]
                    scl_r_enc_i += 1

                scl_ns = [len(scl_left), len(scl_insp), len(scl_right),
                          len(scl_r_minimax)]

                if self._encode_type == "distance":
                    # TODO: decide mdl by linkage
                    left_mdlslcs = MDLSLCS(
                        self._sdm._x[scl_left], labs=[0]*len(scl_left),
                        d=self._sdm._d[scl_left][:, scl_left],
                        metric=self._sdm._metric,
                        encode_type=self._encode_type,
                        mdl_method=self._mdl_method, nprocs=self._nprocs)
                    r_minimax_mdlslcs = MDLSLCS(
                        self._sdm._x[scl_r_minimax],
                        labs=[0]*len(scl_r_minimax),
                        d=self._sdm._d[scl_r_minimax][:, scl_r_minimax],
                        metric=self._sdm._metric,
                        encode_type=self._encode_type,
                        mdl_method=self._mdl_method, nprocs=self._nprocs)
                    left_enc_insp_mdl = left_mdlslcs.encode(
                        self._sdm._d[scl_insp][:, scl_left],
                        col_summary_func=max)
                    r_minimax_enc_insp_mdl = r_minimax_mdlslcs.encode(
                        self._sdm._d[scl_insp][:, scl_r_minimax],
                        col_summary_func=max)
                else:
                    # data
                    # d is not passed
                    left_mdlslcs = MDLSLCS(
                        self._sdm._x[scl_left], labs=[0]*len(scl_left),
                        metric=self._sdm._metric,
                        encode_type=self._encode_type,
                        mdl_method=self._mdl_method, nprocs=self._nprocs)
                    r_minimax_mdlslcs = MDLSLCS(
                        self._sdm._x[scl_r_minimax],
                        labs=[0]*len(scl_r_minimax),
                        metric=self._sdm._metric,
                        encode_type=self._encode_type,
                        mdl_method=self._mdl_method, nprocs=self._nprocs)
                    left_enc_insp_mdl = left_mdlslcs.encode(
                        self._sdm._x[scl_insp], nprocs=self._nprocs)
                    r_minimax_enc_insp_mdl = r_minimax_mdlslcs.encode(
                        self._sdm._x[scl_insp], nprocs=self._nprocs)
                # decide merging direction
                if left_enc_insp_mdl < r_minimax_enc_insp_mdl:
                    # inspected more similar to left
                    scl_left = scl_left + scl_insp
                    scl_insp = scl_right
                    merge_type = "m left"
                else:
                    scl_insp = scl_insp + scl_right
                    merge_type = "m right"

                curr_iter_run_log = str.format(
                    "{}, {} -- sub-cl sizes: {}, "
                    "eval mdl: {}, {}",
                    m_ind, sum(map(len, split_s_inds[:m_ind])), scl_ns,
                    [float(left_enc_insp_mdl), float(r_minimax_enc_insp_mdl),
                     float(left_enc_insp_mdl / r_minimax_enc_insp_mdl)],
                    merge_type)

                self._run_log += curr_iter_run_log
                if self._verbose:
                    print(curr_iter_run_log)
            # insp cluster has >= min_cl_n samples, check whether merge with
            # left or not
            # In this scenario, rhs sub-clusters are non-informative, because
            # we do not know their cluster belongings yet.
            # If we still assume that rhs cluster is just above min_cl_n,
            # we may underestimate the rhs true cluster size, thus causing
            # undesired behavior.
            scl_left_n = np.int_(len(scl_left))
            scl_insp_n = np.int_(len(scl_insp))
            scl_left_ratio = scl_left_n / (scl_left_n + scl_insp_n)
            scl_insp_ratio = scl_insp_n / (scl_left_n + scl_insp_n)
            scl_left_insp = scl_left + scl_insp
            if self._encode_type == "distance":
                l_i_mdl_slcs = MDLSLCS(
                    x=self._sdm._x[scl_left_insp],
                    labs=[0]*scl_left_n + [1]*scl_insp_n,
                    encode_type=self._encode_type, mdl_method=self._mdl_method,
                    d=self._sdm._d[scl_left_insp][:, scl_left_insp],
                    metric=self._sdm._metric, nprocs=self._nprocs)
            else:
                # d is not passed
                l_i_mdl_slcs = MDLSLCS(
                    x=self._sdm._x[scl_left_insp],
                    labs=[0]*scl_left_n + [1]*scl_insp_n,
                    encode_type=self._encode_type, mdl_method=self._mdl_method,
                    metric=self._sdm._metric, nprocs=self._nprocs)
            left_insp_no_lab_mdl = l_i_mdl_slcs.no_lab_mdl(
                nprocs=self._nprocs, verbose=self._verbose)

            left_insp_lab_mdl_res = l_i_mdl_slcs.lab_mdl(
                cl_mdl_scale_factor=self._cl_mdl_scale_factor,
                nprocs=self._nprocs, verbose=self._verbose)
            # TODO: validate ulab_mdls order
            left_split_mdl = left_insp_lab_mdl_res.ulab_mdls[0]
            insp_split_mdl = left_insp_lab_mdl_res.ulab_mdls[1]
            cluster_mdl = left_insp_lab_mdl_res.cluster_mdl
            left_insp_lab_mdl = left_insp_lab_mdl_res.ulab_mdl_sum

            if left_insp_no_lab_mdl < 0:
                min_merge_mdl = ((1 + self._min_split_mdl_red_ratio) *
                                 left_insp_no_lab_mdl)
            else:
                min_merge_mdl = ((1 - self._min_split_mdl_red_ratio) *
                                 left_insp_no_lab_mdl)

            if left_insp_lab_mdl > min_merge_mdl:
                # merge
                merge_type = "merge"
                split_s_inds.insert(m_ind, scl_left + scl_insp)
            else:
                # do not merge
                merge_type = "split"
                split_s_inds.insert(m_ind, scl_insp)
                split_s_inds.insert(m_ind, scl_left)
                m_ind += 1

            curr_iter_run_log = str.format(
                "{}, {} -- no lab mdl: {}, [left, insp] mdl: {}, "
                "cluster_mdl: {}, \n[left, insp] n: {}, "
                "[left, insp] ratio: {}, \n"
                "lab mdl: {}, split/merge: {}, \n"
                "{}.\n\n",
                m_ind,
                sum(map(len, split_s_inds[:m_ind])),
                float(left_insp_no_lab_mdl),
                [float(left_split_mdl), float(insp_split_mdl)],
                cluster_mdl,
                [int(scl_left_n), int(scl_insp_n)],
                [float(scl_left_ratio), float(scl_insp_ratio)],
                float(left_insp_lab_mdl),
                [float(left_insp_lab_mdl / left_insp_no_lab_mdl),
                 float((left_insp_lab_mdl - cluster_mdl) /
                       left_insp_no_lab_mdl),
                 float(left_split_mdl / left_insp_no_lab_mdl),
                 float(insp_split_mdl / left_insp_no_lab_mdl)],
                merge_type)

            self._run_log += curr_iter_run_log
            if self._verbose:
                print(curr_iter_run_log)
        labs = np.concatenate([[i] * len(split_s_inds[i])
                               for i in range(len(split_s_inds))])
        s_inds = np.concatenate(split_s_inds)
        return s_inds, labs
