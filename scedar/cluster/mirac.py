import numpy as np

from .. import eda
from ..eda.slcs import MDLSingleLabelClassifiedSamples as MDLSLCS
from .. import utils


class MIRAC(object):
    """
    MIRAC: MDL iteratively regularized agglomerative clustering.

    Args:
        x (float array): Data matrix.
        d (float array): Distance matrix.
        metric (str): Type of distance metric.
        sids (sid list): List of sample ids.
        fids (fid list): List of feature ids.
        hac_tree (HCTree): Hierarchical tree built by agglomerative clustering
            to divide in MIRAC. If provided, distance matrix will not be used
            for building another tree.
        nprocs (int): Number of processes to run MIRAC parallely.
        cl_mdl_scale_factor (float): Scale factor of cluster overhead mdl.
        min_cl_n (int): Minimum # samples in a cluster.
        encode_type ("auto", "data", or "distance"): Type of values to encode.
            If "auto", encode data when n_features <= 100.
        mdl_method (mdl.Mdl): If None, use ZeroIGKdeMdl for encoded values
            with >= 50% zeros, and use GKdeMdl otherwise.
        linkage (str): Linkage type for generating the hierarchy.
        verbose (bool): Print stats for each iteration.

    Attributes:
        _sdm (SampleDistanceMatrix): Data and distance matrices.
        _min_cl_n (int): Stored parameter.
        _encode_type (str): Encode type. If "auto" provided, this attribute
            will store the determined encode type.
        _mdl_method (mdl.Mdl): Mdl method. If None is provided, this attribute
            will store the determined mdl method.
        labs (label list): Labels of clustered samples. 1-to-1 matching to
            from first to last.
        _hac_tree (eda.hct.HClustTree): Root node of the hierarchical
            agglomerative clustering tree.
        _run_log (str): String containing the log of the MIRAC run.

    TODO:
    * Dendrogram representation of the splitting process.

    * Take HCTree as parameter. Computing it is non-trivial when n is large.

    * Simplify splitting criteria.
    """

    def __init__(self, x, d=None, metric=None, sids=None, fids=None,
                 hac_tree=None, nprocs=1, cl_mdl_scale_factor=1,
                 min_cl_n=25, encode_type="auto", mdl_method=None,
                 min_split_mdl_red_ratio=0.2,
                 linkage="complete", optimal_ordering=True,
                 verbose=False):
        super().__init__()
        # initialize simple attributes
        self._nprocs = max(int(nprocs), 1)
        self._is_euc_dist = metric == "euclidean"
        self._verbose = verbose
        self._linkage = linkage
        self._optimal_ordering = optimal_ordering
        self._min_split_mdl_red_ratio = min_split_mdl_red_ratio
        self._sdm = eda.SampleDistanceMatrix(x=x, d=d, metric=metric,
                                             sids=sids, fids=fids,
                                             nprocs=nprocs)
        # initialize encode type
        if encode_type not in ("auto", "data", "distance"):
            raise ValueError("encode_type must in "
                             "('auto', 'data', 'distance')."
                             "Provided: {}".format(encode_type))
        if encode_type == "auto":
            if self._sdm._x.shape[1] > 100:
                encode_type = "distance"
            else:
                encode_type = "data"
        self._encode_type = encode_type
        # initialize mdl method
        if mdl_method is None:
            if encode_type == "data":
                ex = self._sdm._x
            else:
                ex = self._sdm._d
            if ex.size == 0:
                # empty matrix
                mdl_method = eda.mdl.GKdeMdl
            else:
                n_nonzero = np.count_nonzero(ex)
                if n_nonzero / ex.size > 0.5:
                    mdl_method = eda.mdl.GKdeMdl
                else:
                    mdl_method = eda.mdl.ZeroIGKdeMdl
        self._mdl_method = mdl_method
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
        # run MIRAC with initialized parameters
        s_inds, s_labs = self._mirac()
        # Initialize labels
        self._labs = s_labs[np.argsort(s_inds, kind="mergesort")].tolist()

    @property
    def labs(self):
        return self._labs.copy()

    @staticmethod
    def _encode_dmat(dmat, fit_s_inds, q_s_inds, mdl_method, nprocs=1):
        """Private method to encode distance matrix
        """
        def single_fit_s_enc_mdl(i):
            # copy indices for parallel processing
            i_s_ind = fit_s_inds[i]
            non_i_s_inds = fit_s_inds[:i] + fit_s_inds[i+1:]
            i_encoder = mdl_method(dmat[i_s_ind, non_i_s_inds])
            i_encode_q_mdl = i_encoder.encode(dmat[i_s_ind, q_s_inds])
            return i_encode_q_mdl

        n_fit = len(fit_s_inds)
        fit_encode_q_mdls = utils.parmap(single_fit_s_enc_mdl,
                                         range(n_fit), nprocs=nprocs)
        return fit_encode_q_mdls

    @staticmethod
    def _dmat_mdl(dmat, mdl_method, nprocs=1):
        """Private method to compute mdl for distance matrix
        """
        def single_s_mdl(i):
            # copy indices for parallel processing
            i_s_ind = fit_s_inds[i]
            non_i_s_inds = fit_s_inds[:i] + fit_s_inds[i+1:]
            return mdl_method(dmat[i_s_ind, non_i_s_inds]).mdl
        n = dmat.shape[0]
        dmat_ind_mdls = utils.parmap(single_s_mdl, range(n_fit), nprocs=nprocs)
        dmat_mdl = np.sum(dmat_ind_mdls)
        return dmat_mdl

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
                    soft_min_subtree_size=1,
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
                    left_enc_insp_mdl = np.max(self._encode_dmat(
                        dmat=self._sdm._d, fit_s_inds=scl_left,
                        q_s_inds=scl_insp, mdl_method=self._mdl_method,
                        nprocs=self._nprocs))
                    r_minimax_enc_insp_mdl = np.max(self._encode_dmat(
                        dmat=self._sdm._d, fit_s_inds=scl_r_minimax,
                        q_s_inds=scl_insp, mdl_method=self._mdl_method,
                        nprocs=self._nprocs))
                else:
                    # data
                    left_enc_insp_mdl = MDLSLCS(
                        self._sdm._x[scl_left], [0]*len(scl_left),
                        mdl_method=self._mdl_method, metric=self._sdm._metric,
                        nprocs=self._nprocs).encode(self._sdm._x[scl_insp],
                                                    nprocs=self._nprocs)
                    r_minimax_enc_insp_mdl = MDLSLCS(
                        self._sdm._x[scl_r_minimax], [0]*len(scl_r_minimax),
                        mdl_method=self._mdl_method, metric=self._sdm._metric,
                        nprocs=self._nprocs).encode(self._sdm._x[scl_insp],
                                                    nprocs=self._nprocs)
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
            if self._encode_type == "distance":
                left_mdl = self._dmat_mdl(
                    self._sdm._d[scl_left][:, scl_left], self._mdl_method,
                    nprocs=self._nprocs)
                insp_mdl = self._dmat_mdl(
                    self._sdm._d[scl_insp][:, scl_insp], self._mdl_method,
                    nprocs=self._nprocs)
                cluster_mdl = MDLSLCS.compute_cluster_mdl(
                    [0]*scl_left_n + [1]*scl_insp_n,
                    cl_mdl_scale_factor=self._cl_mdl_scale_factor)

                left_split_mdl = scl_left_ratio * cluster_mdl + left_mdl
                insp_split_mdl = scl_insp_ratio * cluster_mdl + insp_mdl

                left_insp_no_lab_mdl = self._dmat_mdl(
                    self._sdm._d[scl_left + scl_insp][:, scl_left + scl_insp],
                    self._mdl_method, nprocs=self._nprosc)
            else:
                # encode_type == "data"
                # left and inspected SLCS
                l_i_mdl_slcs = MDLSLCS(
                    x=self._sdm._x[scl_left + scl_insp],
                    labs=[0]*len(scl_left) + [1]*len(scl_insp),
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

            left_insp_lab_mdl = left_split_mdl + insp_split_mdl

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
