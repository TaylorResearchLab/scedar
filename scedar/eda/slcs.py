import numpy as np

import matplotlib as mpl
import matplotlib.colors
import seaborn as sns

from .sdm import SampleDistanceMatrix
from . import mtype

class SingleLabelClassifiedSamples(SampleDistanceMatrix):
    """SingleLabelClassifiedSamples"""
    # sid, lab, fid, x

    def __init__(self, x, labs, sids=None, fids=None, d=None,
                 metric="correlation", nprocs=None):
        # sids: sample IDs. String or int.
        # labs: sample classified labels. String or int.
        # x: (n_samples, n_features)
        super(SingleLabelClassifiedSamples, self).__init__(x=x, d=d,
                                                           metric=metric,
                                                           sids=sids, fids=fids,
                                                           nprocs=nprocs)
        mtype.check_is_valid_labs(labs)
        labs = np.array(labs)
        if self._sids.shape[0] != labs.shape[0]:
            raise ValueError("sids must have the same length as labs")
        self._labs = labs

        self._uniq_labs, self._uniq_lab_cnts = np.unique(labs,
                                                         return_counts=True)

        sid_lut = {}
        for uniq_lab in self._uniq_labs:
            sid_lut[uniq_lab] = self._sids[labs == uniq_lab]
        self._sid_lut = sid_lut

        lab_lut = {}
        # sids only contain unique values
        for i in range(self._sids.shape[0]):
            lab_lut[self._sids[i]] = labs[i]
        self._lab_lut = lab_lut
        return

    def filter_min_class_n(self, min_class_n):
        uniq_lab_cnts = np.unique(self._labs, return_counts=True)
        nf_sid_ind = np.in1d(
            self._labs, (uniq_lab_cnts[0])[uniq_lab_cnts[1] >= min_class_n])
        return self.ind_x(nf_sid_ind)

    def labs_to_sids(self, labs):
        return tuple(tuple(self._sid_lut[y].tolist()) for y in labs)

    def sids_to_labs(self, sids):
        return np.array([self._lab_lut[x] for x in sids])

    def ind_x(self, selected_s_inds=None, selected_f_inds=None):
        """
        Subset samples by (sample IDs, feature IDs).
        
        Parameters
        ----------
        selected_s_inds: int array
            Index array of selected samples. If is None, select all.
        selected_f_inds: int array
            Index array of selected features. If is None, select all.

        Returns
        -------
        subset: SingleLabelClassifiedSamples
        """
        if selected_s_inds is None:
            selected_s_inds = slice(None, None)

        if selected_f_inds is None:
            selected_f_inds = slice(None, None)

        return SingleLabelClassifiedSamples(
            x=self._x[selected_s_inds, :][:, selected_f_inds].copy(),
            labs=self._labs[selected_s_inds].tolist(),
            d=self._d[selected_s_inds, :][:, selected_s_inds].copy(),
            sids=self._sids[selected_s_inds].tolist(),
            fids=self._fids[selected_f_inds].tolist(),
            metric=self._metric, nprocs=self._nprocs)

    def id_x(self, selected_sids=None, selected_fids=None):
        """
        Subset samples by (sample IDs, feature IDs).
        
        Parameters
        ----------
        selected_s_inds: int array
            Index array of selected samples. If is None, select all.
        selected_f_inds: int array
            Index array of selected features. If is None, select all.

        Returns
        -------
        subset: SingleLabelClassifiedSamples
        """
        if selected_sids is None:
            selected_s_inds = None
        else:
            selected_s_inds = self.s_id_to_ind(selected_sids)

        if selected_fids is None:
            selected_f_inds = None
        else:
            selected_f_inds = self.f_id_to_ind(selected_fids)
        return self.ind_x(selected_s_inds, selected_f_inds)

    def tsne_gradient_plot(self, gradient=None, labels=None, 
                           title=None, xlab=None, ylab=None,
                           figsize=(20, 20), add_legend=True, 
                           n_txt_per_cluster=3, alpha=1, s=0.5,
                           random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.
        """
        return super(SingleLabelClassifiedSamples, 
                     self).tsne_gradient_plot(
                        labels=self.labs, gradient=gradient,
                        title=title, xlab=xlab, ylab=ylab,
                        figsize=figsize, 
                        add_legend=add_legend, 
                        n_txt_per_cluster=n_txt_per_cluster, 
                        alpha=alpha, s=s, 
                        random_state=random_state, 
                        **kwargs)

    def tsne_feature_gradient_plot(self, fid, labels=None, 
                                   title=None, xlab=None, ylab=None,
                                   figsize=(20, 20), add_legend=True, 
                                   n_txt_per_cluster=3, alpha=1, s=0.5,
                                   random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.
        """
        return super(SingleLabelClassifiedSamples, 
                     self).tsne_feature_gradient_plot(
                        fid=fid, labels=self.labs,
                        title=title, xlab=xlab, ylab=ylab,
                        figsize=figsize, 
                        add_legend=add_legend, 
                        n_txt_per_cluster=n_txt_per_cluster, 
                        alpha=alpha, s=s, 
                        random_state=random_state, 
                        **kwargs)

    @property
    def labs(self):
        return self._labs.tolist()

    # Sort the clustered sample_ids with the reference order of another.
    #
    # Sort sids according to labs
    # If ref_sid_order is not None:
    #   sort sids further according to ref_sid_order
    def lab_sorted_sids(self, ref_sid_order=None):
        sep_lab_sid_list = []
        sep_lab_list = []
        for iter_lab in sorted(self._sid_lut.keys()):
            iter_sid_arr = self._sid_lut[iter_lab]
            sep_lab_sid_list.append(iter_sid_arr)
            sep_lab_list.append(np.repeat(iter_lab, len(iter_sid_arr)))

        if ref_sid_order is not None:
            mtype.check_is_valid_sfids(ref_sid_order)
            ref_sid_order = np.array(ref_sid_order)
            # sort r according to q
            # assumes:
            # - r contains all elements in q
            # - r is 1d np array

            def sort_flat_sids(query_sids, ref_sids):
                return ref_sids[np.in1d(ref_sids, query_sids)]

            # sort inner sid list but maintains the order as sep_lab_list
            sep_lab_sid_list = [sort_flat_sids(x, ref_sid_order)
                                for x in sep_lab_sid_list]
            sep_lab_min_sid_list = [x[0] for x in sep_lab_sid_list]
            sorted_sep_lab_min_sid_list = list(
                sort_flat_sids(sep_lab_min_sid_list, ref_sid_order))
            min_sid_sorted_sep_lab_ind_list = [sep_lab_min_sid_list.index(x)
                                               for x in sorted_sep_lab_min_sid_list]
            sep_lab_list = [sep_lab_list[i]
                            for i in min_sid_sorted_sep_lab_ind_list]
            sep_lab_sid_list = [sep_lab_sid_list[i]
                                for i in min_sid_sorted_sep_lab_ind_list]

        lab_sorted_sid_arr = np.concatenate(sep_lab_sid_list)
        lab_sorted_lab_arr = np.concatenate(sep_lab_list)

        # check sorted sids are the same set as original
        np.testing.assert_array_equal(
            np.sort(lab_sorted_sid_arr), np.sort(self._sids))
        # check sorted labs are the same set as original
        np.testing.assert_array_equal(
            np.sort(lab_sorted_lab_arr), np.sort(self._labs))
        # check sorted (sid, lab) matchings are the same set as original
        np.testing.assert_array_equal(lab_sorted_lab_arr[np.argsort(lab_sorted_sid_arr)],
                                      self._labs[np.argsort(self._sids)])

        return (lab_sorted_sid_arr, lab_sorted_lab_arr)

    # See how two clustering criteria match with each other.
    # When given q_slc_samples is not None, sids and labs are ignored.
    # When q_slc_samples is None, sids and labs must be provided
    def cross_labs(self, q_slc_samples):
        if not isinstance(q_slc_samples, SingleLabelClassifiedSamples):
            raise TypeError("Query should be an instance of "
                            "SingleLabelClassifiedSamples")

        try:
            ref_labs = np.array([self._lab_lut[x]
                                 for x in q_slc_samples.sids])
        except KeyError as e:
            raise ValueError("query sid {} is not in ref sids.".format(e))

        query_labs = np.array(q_slc_samples.labs)

        uniq_rlabs, uniq_rlab_cnts = np.unique(ref_labs, return_counts=True)
        cross_lab_lut = {}
        for i in range(len(uniq_rlabs)):
            # ref cluster i. query unique labs.
            ref_ci_quniq = tuple(map(list, np.unique(
                query_labs[np.where(np.array(ref_labs) == uniq_rlabs[i])],
                return_counts=True)))
            cross_lab_lut[uniq_rlabs[i]] = (uniq_rlab_cnts[i],
                                            tuple(map(tuple, ref_ci_quniq)))

        return cross_lab_lut
