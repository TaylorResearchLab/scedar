import numpy as np

from scedar.eda import SampleDistanceMatrix
from scedar.eda.slcs import SingleLabelClassifiedSamples as SLCS
from scedar import utils

import leidenalg as la

class Community(object):
    """
    Community clustering

    Args
    ----
    x : float array
        Data matrix.
    d : float array
        Distance matrix.
    graph: igraph.Graph
        Need to have a weight attribute as affinity. If this argument
        is not None, the graph will directly be used for community
        clustering.
    metric: {'cosine', 'euclidean'}
        Metric used for nearest neighbor computation.
    sids : sid list
        List of sample ids.
    fids : fid list
        List of feature ids.
    use_pdist : boolean
        To use the pairwise distance matrix or not. The pairwise distance
        matrix may be too large to save for datasets with a large number of
        cells.
    k : int
        The number of nearest neighbors.
    use_pca : bool
        Use PCA for nearest neighbors or not.
    use_hnsw : bool
        Use Hierarchical Navigable Small World graph to compute
        approximate nearest neighbor.
    index_params : dict
        Parameters used by HNSW in indexing.

        efConstruction : int
            Default 100. Higher value improves the quality of a constructed
            graph and leads to higher accuracy of search. However this also
            leads to longer indexing times. The reasonable range of values
            is 100-2000.
        M : int
            Default 5. Higher value leads to better recall and shorter
            retrieval times, at the expense of longer indexing time. The
            reasonable range of values is 5-100.
        delaunay_type : {0, 1, 2, 3}
            Default 2. Pruning heuristic, which affects the trade-off
            between retrieval performance and indexing time. The default
            is usually quite good.
        post : {0, 1, 2}
            Default 0. The amount and type of postprocessing applied to the
            constructed graph. 0 means no processing. 2 means more
            processing.
        indexThreadQty : int
            Default self._nprocs. The number of threads used.

    query_params : dict
        Parameters used by HNSW in querying.

        efSearch : int
            Default 100. Higher value improves recall at the expense of
            longer retrieval time. The reasonable range of values is
            100-2000.

    aff_scale : float > 0
        Scaling factor used for converting distance to affinity.
        Affinity = (max(distance) - distance) * aff_scale.
    partition_method : str
        Following methods are implemented in leidenalg package:

        - RBConfigurationVertexPartition: only well-defined for positive edge
          weights.
        - RBERVertexPartition: well-defined only for positive edge weights.
        - CPMVertexPartition: well-defined for both positive and negative edge
          weights.
        - SignificanceVertexPartition: well-defined only for unweighted graphs.
        - SurpriseVertexPartition: well-defined only for positive edge weights.
    resolution : float > 0
        Resolution used for community clustering. Higer value produces more
        clusters.
    random_state : int
        Random number generator seed used for community clustering.
    n_iter : int
        Number of iterations used for community clustering.
    nprocs : int > 0
        The number of processes/cores used for community clustering.
    verbose : bool
        Print progress or not.

    Attributes
    ----------
        labs : label list
            Labels of clustered samples. 1-to-1 matching to
            from first to last.
        _sdm : SampleDistanceMatrix
            Data and distance matrices.
        _graph : igraph.Graph
            Graph used for clustering.
        _la_res : leidenalg.VertexPartition
            Partition results computed by leidenalg.
        _k
        _use_pca
        _use_hnsw
        _index_params
        _query_params
        _aff_scale
    """

    def __init__(self, x, d=None, graph=None,
                 metric="cosine", sids=None, fids=None,
                 use_pdist=False,  k=15, use_pca=True, use_hnsw=True,
                 index_params=None, query_params=None, aff_scale=1,
                 partition_method="RBConfigurationVertexPartition",
                 resolution=1, random_state=None, n_iter=2,
                 nprocs=1, verbose=False):
        super().__init__()
        if aff_scale <= 0:
            raise ValueError("Affinity scaling (aff_scale) shoud > 0.")

        if metric not in ("cosine", "euclidean"):
            raise ValueError("Metric only supports cosine and euclidean.")

        self._sdm = SampleDistanceMatrix(x=x, d=d, metric=metric,
                                         use_pdist=use_pdist,
                                         sids=sids, fids=fids, nprocs=nprocs)
        if graph is None:
            knn_conn_mat = self._sdm.s_knn_connectivity_matrix(
                k=k, use_pca=use_pca, use_hnsw=use_hnsw,
                index_params=index_params, query_params=query_params,
                verbose=verbose)
            graph = SampleDistanceMatrix.knn_conn_mat_to_aff_graph(
                knn_conn_mat, aff_scale=aff_scale)

        if partition_method == "RBConfigurationVertexPartition":
            la_part_cls = la.RBConfigurationVertexPartition
        elif partition_method == "RBERVertexPartition":
            la_part_cls = la.RBERVertexPartition
        elif partition_method == "CPMVertexPartition":
            la_part_cls = la.CPMVertexPartition
        elif partition_method == "SignificanceVertexPartition":
            la_part_cls = la.SignificanceVertexPartition
        elif partition_method == "SurpriseVertexPartition":
            la_part_cls = la.SurpriseVertexPartition
        else:
            raise ValueError(
                "Unknown partition method: {}".format(partition_method))

        la_res = la.find_partition(graph, la.RBConfigurationVertexPartition,
                                   seed=random_state, weights='weight',
                                   resolution_parameter=resolution)
        # keep track of results and parameters
        self._graph = graph
        self._la_res = la_res
        self._labs = la_res.membership
        self._k = k
        self._use_pca = use_pca
        self._use_hnsw = use_hnsw
        self._index_params = index_params
        self._query_params = query_params
        self._aff_scale = aff_scale

    @property
    def labs(self):
        return self._labs.copy()
