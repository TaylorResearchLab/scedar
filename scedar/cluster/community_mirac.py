import numpy as np

from collections import defaultdict

from scedar.cluster import MIRAC
from scedar.cluster import Community


class CommunityMIRAC(object):
    """
    CommunityMIRAC: Community + MIRAC clustering

    Run community clustering with high resolution to get a large number of
    clusters. Then, run MIRAC on the community clusters.

    Args
    ----
    x : float array
        Data matrix.
    d : float array
        Distance matrix.
    sids : sid list
        List of sample ids.
    fids : fid list
        List of feature ids.
    nprocs : int > 0
        The number of processes/cores used for community clustering.
    verbose : bool
        Print progress or not.

    Attributes
    ----------
    _x : float array
        Data matrix.
    _d : float array
        Distance matrix.
    _sids : sid list
        List of sample ids.
    _fids : fid list
        List of feature ids.
    _nprocs : int > 0
        The number of processes/cores used for community clustering.
    _verbose : bool
        Print progress or not.
    _cm_res : cluster.Community
        Community clustering result.
    _cm_clp_x : array
        Data array with samples collapsed by community clustering labels.
        For each cluster, the mean of all samples is a row in this array.
    _mirac_res : cluster.MIRAC
        MIRAC clustering results on _cm_clp_x
    labs : list
        list of labels
    """

    def __init__(self, x, d=None, sids=None, fids=None,
                 nprocs=1, verbose=False):
        super().__init__()
        self._x = x
        self._d = d
        self._sids = sids
        self._fids = fids
        self._nprocs = nprocs
        self._verbose = verbose
        self._cm_res = None
        self._cm_clp_x = None
        self._mirac_res = None
        self._labs = None

    def run_community(self, graph=None, metric="cosine",
                      use_pdist=False,  k=15, use_pca=True, use_hnsw=True,
                      index_params=None, query_params=None, aff_scale=1,
                      partition_method="RBConfigurationVertexPartition",
                      resolution=100, random_state=None, n_iter=2,
                      nprocs=None):
        if nprocs is None:
            nprocs = self._nprocs
        self._cm_res = Community(x=self._x, d=self._d, graph=graph,
                                 metric=metric, sids=self._sids,
                                 fids=self._fids,
                                 use_pdist=use_pdist, k=k, use_pca=use_pca,
                                 use_hnsw=use_hnsw, index_params=index_params,
                                 query_params=query_params,
                                 aff_scale=aff_scale,
                                 partition_method=partition_method,
                                 resolution=resolution,
                                 random_state=random_state,
                                 n_iter=n_iter, nprocs=nprocs,
                                 verbose=self._verbose)
        if self._verbose:
            print("Community cluster: {}".format(
                self._cm_res._la_res.summary()))

        self._cm_clp_x = self.collapse_clusters(self._x, self._cm_res.labs)


    def run_mirac(self, metric="cosine", hac_tree=None, cl_mdl_scale_factor=1,
                  min_cl_n=25, encode_type="auto", mdl_method=None,
                  min_split_mdl_red_ratio=0.2, soft_min_subtree_size=1,
                  linkage="complete", optimal_ordering=False,
                  dim_reduct_method=None, nprocs=None):
        if self._cm_clp_x is None:
            raise ValueError("Need to run community clustering first.")

        if nprocs is None:
            nprocs = self._nprocs

        self._mirac_res = MIRAC(
            self._cm_clp_x, metric=metric,
            sids=self._sids, fids=self._fids,
            hac_tree=hac_tree, nprocs=nprocs,
            cl_mdl_scale_factor=cl_mdl_scale_factor,
            min_cl_n=min_cl_n, encode_type=encode_type,
            mdl_method=mdl_method,
            min_split_mdl_red_ratio=min_split_mdl_red_ratio,
            soft_min_subtree_size=soft_min_subtree_size,
            linkage=linkage, optimal_ordering=optimal_ordering,
            dim_reduct_method=dim_reduct_method,
            verbose=self._verbose)

        self._merge_labels()

    def _merge_labels(self):
        l1_cm_labs = self._cm_res.labs
        l2_mirac_labs = self._mirac_res.labs
        self._labs = [l2_mirac_labs[i] for i in l1_cm_labs]

    def tune_mirac(self, cl_mdl_scale_factor=1, min_cl_n=25,
                   min_split_mdl_red_ratio=0.2,
                   soft_min_subtree_size=1, verbose=False):
        if self._mirac_res is None:
            raise ValueError("Need to run MIRAC first.")

        self._mirac_res.tune_parameters(cl_mdl_scale_factor, min_cl_n,
                                        min_split_mdl_red_ratio,
                                        soft_min_subtree_size, verbose)

        self._merge_labels()


    def run(self, graph=None, metric="cosine",
            use_pdist=False,  k=15, use_pca=True, use_hnsw=True,
            index_params=None, query_params=None, aff_scale=1,
            partition_method="RBConfigurationVertexPartition",
            resolution=100, random_state=None, n_iter=2,
            hac_tree=None, cl_mdl_scale_factor=1,
            min_cl_n=25, encode_type="auto", mdl_method=None,
            min_split_mdl_red_ratio=0.2,
            soft_min_subtree_size=1,
            linkage="complete", optimal_ordering=False, nprocs=None):

        self.run_community(
            graph=graph, metric=metric,
            use_pdist=use_pdist,  k=k, use_pca=use_pca, use_hnsw=use_hnsw,
            index_params=index_params, query_params=query_params,
            aff_scale=aff_scale,
            partition_method=partition_method, nprocs=nprocs,
            resolution=resolution, random_state=random_state, n_iter=n_iter)

        self.run_mirac(
            metric=metric, hac_tree=hac_tree,
            cl_mdl_scale_factor=cl_mdl_scale_factor,
            min_cl_n=min_cl_n, encode_type=encode_type,
            mdl_method=mdl_method,
            min_split_mdl_red_ratio=min_split_mdl_red_ratio,
            soft_min_subtree_size=soft_min_subtree_size, nprocs=nprocs,
            linkage=linkage, optimal_ordering=optimal_ordering)


    @staticmethod
    def collapse_clusters(data_x, cluster_labs):
        uniq_labs = sorted(set(cluster_labs))
        if uniq_labs != list(range(len(uniq_labs))):
            raise ValueError("labels must be integers from 0 to the number"
                             "of clusters. There should be no missing ones.")

        cl_lab_sinds_lut = defaultdict(list)
        for s_ind, s_lab in enumerate(cluster_labs):
            cl_lab_sinds_lut[s_lab].append(s_ind)

        cl_stat_vecs = []
        # need to be sorted
        for lab in uniq_labs:
            s_inds = cl_lab_sinds_lut[lab]
            s_inds_x_mean = data_x[s_inds, :].mean(axis=0)
            if s_inds_x_mean.ndim == 2:
                # matrix behavior
                assert s_inds_x_mean.shape[0] == 1
                s_inds_x_mean = s_inds_x_mean.A1
            cl_stat_vecs.append(s_inds_x_mean)
        return np.vstack(cl_stat_vecs)

    @property
    def labs(self):
        return self._labs.copy()
