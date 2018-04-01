import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.utils

import matplotlib as mpl
import matplotlib.colors
import seaborn as sns

from collections import defaultdict

import xgboost as xgb

from .sdm import SampleDistanceMatrix
from .plot import swarm
from . import mtype
from .. import utils


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

        self._xgb_lut = {}
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

    def lab_x(self, selected_labs):
        if selected_labs is None:
            raise ValueError("selected_labs should be a non-empty list.")

        if np.isscalar(selected_labs):
            selected_labs = [selected_labs]

        if not all([x in self._uniq_labs.tolist() for x in selected_labs]):
            raise ValueError("selected_labs: {} are not all existed "
                             "in the SLCS unique labels "
                             "{}".format(selected_labs, self._uniq_labs))

        lab_selected_s_bool_arr = np.in1d(self._labs, selected_labs)
        return self.ind_x(lab_selected_s_bool_arr)

    @staticmethod
    def _xgb_train_runner(x, lab_inds, str_fids, test_size=0.3,
                          num_boost_round=10, nprocs=1,
                          random_state=None, silent=1, xgb_params=None):
        """
        Run xgboost train with prepared data and parameters.

        Returns
        -------
        sorted_fscore_list: list
            Ordered important features and their results.
        bst: xgb.Booster
            Fitted model.
        eval_stats: tuple
            Final test and train error.
        """
        n_uniq_labs = len(np.unique(lab_inds))
        # Prepare default xgboost parameters
        xgb_random_state = 0 if random_state is None else random_state
        if xgb_params is None:
            # Use log2(n_features) as default depth.
            xgb_params = {
                "eta": 0.3,
                "max_depth": 6,
                "silent": silent,
                "nthread": nprocs,
                "alpha": 1,
                "lambda": 0,
                "seed": xgb_random_state
            }
            if n_uniq_labs == 2:
                # do binary classification
                xgb_params["objective"] = "binary:logistic"
                xgb_params["eval_metric"] = "error"
            else:
                # do multi-label classification
                xgb_params["num_class"] = n_uniq_labs
                xgb_params["objective"] = "multi:softmax"
                xgb_params["eval_metric"] = "merror"
        # split training and testing
        # random state determined by numpy
        train_x, test_x, train_labs, test_labs = train_test_split(
            x, lab_inds, test_size=test_size)
        # xgb datastructure to hold data and labels
        dtrain = xgb.DMatrix(train_x, train_labs, feature_names=str_fids)
        dtest = xgb.DMatrix(test_x, test_labs, feature_names=str_fids)
        # list of data to evaluate
        eval_list = [(dtest, "test"), (dtrain, "train")]
        evals_result = {}
        if silent:
            verbose_eval = False
        else:
            verbose_eval = True
        # bst is the train boost tree model
        bst = xgb.train(xgb_params, dtrain, num_boost_round, eval_list,
                        evals_result=evals_result, verbose_eval=verbose_eval)
        # Turn dict to list
        # [ [('train...', float), ...],
        #   [('test...', float), ...] ]
        eval_stats = [ [(eval_name + " " + mname,
                         mval_list[num_boost_round-1])
                        for mname, mval_list in eval_dict.items()]
                       for eval_name, eval_dict in evals_result.items() ]
        # {feature_name: fscore, ...}
        fscore_dict = bst.get_fscore()
        sorted_fscore_list = sorted(fscore_dict.items(), key=lambda t: t[1],
                                    reverse=True)
        return sorted_fscore_list, bst, eval_stats

    def feature_importance_across_labs(self, selected_labs, test_size=0.3,
                                       num_boost_round=10, nprocs=1,
                                       random_state=None, silent=1,
                                       xgb_params=None,
                                       num_bootstrap_round=0,
                                       bootstrap_size=None,
                                       shuffle_features=False):
        """
        Use xgboost to determine the importance of features determining the
        difference between samples with different labels.

        Run cross validation on dataset and obtain import features.

        Parameters
        ----------
        selected_labs: label list
            Labels to compare using xgboost.
        test_size: float in range (0, 1)
            Ratio of samples to be used for testing
        num_bootstrap_round: int
            Do num_bootstrap_round times of simple bootstrapping if
            `num_bootstrap_round > 0`.
        bootstrap_size: int
            The number of samples for each bootstrapping run.
        shuffle_features: bool
        num_boost_round: int
            The number of rounds for xgboost training.
        random_state: int
        nprocs: int
        xgb_params: dict
            Parameters for xgboost run. If None, default will be used. If
            provided, they will be directly used for xgbooster.

        Returns
        -------
        feature_importance_list: list of feature importance of each run
            `[(feature_id, fscore), ...]`
        bst_list: list of xgb Booster
            Fitted boost tree model

        Notes
        -----
        If multiple features are highly correlated, they may not all show up
        in the resulting tree. You could try to reduce redundant features first
        before comparing different clusters, or you could also interpret the
        important features further after obtaining the important features.

        For details about xgboost parameters, check the following links:

        [1] https://www.analyticsvidhya.com/blog/2016/03/\
complete-guide-parameter-tuning-xgboost-with-codes-python/

        [2] http://xgboost.readthedocs.io/en/latest/python/python_intro.html

        [3] http://xgboost.readthedocs.io/en/latest/parameter.html

        [4] https://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html

        [5] https://www.analyticsvidhya.com/blog/2016/03/\
complete-guide-parameter-tuning-xgboost-with-codes-python/
        """
        num_boost_round = int(num_boost_round)
        if num_boost_round <= 0:
            raise ValueError("num_boost_round must >= 1")
        # This is for implementing caching in the future.
        selected_uniq_labs = np.unique(selected_labs).tolist()
        # subset SLCS
        lab_selected_slcs = self.lab_x(selected_uniq_labs)
        # unique labels in SLCS after subsetting
        # Since lab_x checks whether selected labels are all existing,
        # the unique labels of the subset is equivalent to input selected
        # labels.
        uniq_labs = lab_selected_slcs._uniq_labs.tolist()
        # convert labels into indices from 0 to n_classes
        n_uniq_labs = len(uniq_labs)
        if n_uniq_labs <= 1:
            raise ValueError("The number of unique labels should > 1. "
                             "Provided uniq labs:"
                             " {}".format(uniq_labs))
        lab_ind_lut = dict(zip(uniq_labs, range(n_uniq_labs)))
        lab_inds = [lab_ind_lut[lab] for lab in lab_selected_slcs._labs]

        np.random.seed(random_state)
        # shuffle features if necessary
        str_fids = list(map(str, lab_selected_slcs._fids))
        if shuffle_features:
            feature_inds = np.arange(lab_selected_slcs._x.shape[1])
            feature_inds, str_fids = sklearn.utils.shuffle(
                feature_inds, str_fids)
        else:
            feature_inds = slice(None, None)
        # perform bootstrapping if necessary
        num_bootstrap_round = int(num_bootstrap_round)
        if num_bootstrap_round <= 0:
            # no bootstrapping
            # _xgb_train_runner returns (fscores, bst, eval_stats)
            fscores, bst, eval_stats = self._xgb_train_runner(
                lab_selected_slcs._x[:, feature_inds],
                lab_inds, str_fids, test_size=test_size,
                num_boost_round=num_boost_round,
                xgb_params=xgb_params, random_state=random_state,
                nprocs=nprocs, silent=silent)
            print(eval_stats)
            return fscores, [bst]
        else:
            # do bootstrapping
            # ([dict of scores], [list of bsts], dict of eval stats)
            fs_dict = defaultdict(lambda : 0)
            bst_list = []
            eval_stats_dict = defaultdict(list)
            if bootstrap_size is None:
                bootstrap_size = lab_selected_slcs._x.shape[0]
            sample_inds = np.arange(lab_selected_slcs._x.shape[0])
            # bootstrapping rounds
            for i in range(num_bootstrap_round):
                # random state determined by numpy
                # ensure all labels present
                # initialize resample sample_indices and labels
                bs_s_inds, bs_lab_inds = sklearn.utils.resample(
                    sample_inds, lab_inds, replace=True,
                    n_samples=bootstrap_size)
                while len(np.unique(bs_lab_inds)) != n_uniq_labs:
                    bs_s_inds, bs_lab_inds = sklearn.utils.resample(
                        sample_inds, lab_inds, replace=True,
                        n_samples=bootstrap_size)
                fscores, bst, eval_stats = self._xgb_train_runner(
                    lab_selected_slcs._x[bs_s_inds, :][:, feature_inds],
                    bs_lab_inds, str_fids, test_size=test_size,
                    num_boost_round=num_boost_round,
                    xgb_params=xgb_params, random_state=random_state,
                    nprocs=nprocs, silent=silent)
                # Sum fscores
                for fid, fs in fscores:
                    fs_dict[fid] += fs
                bst_list.append(bst)
                # est: eval stats tuple
                # [ [('train...', float), ...],
                #   [('test...', float), ...] ]
                for elist in eval_stats:
                    for ename, evalue in elist:
                        eval_stats_dict[ename].append(evalue)
                if shuffle_features:
                    feature_inds, str_fids = sklearn.utils.shuffle(
                        feature_inds, str_fids)
            # average score
            for fid in fs_dict:
                fs_dict[fid] /= num_bootstrap_round
            sorted_fs_list = sorted(fs_dict.items(), key=lambda t: t[1],
                                    reverse=True)
            # calculate mean +/- std of eval stats
            for ename, evalue_list in eval_stats_dict.items():
                print("{}: mean {}, std {}".format(
                    ename, np.mean(evalue_list), np.std(evalue_list, ddof=1)))
            return sorted_fs_list, bst_list

    def tsne_gradient_plot(self, gradient=None, labels=None,
                           selected_labels=None,
                           shuffle_label_colors=False,
                           title=None, xlab=None, ylab=None,
                           figsize=(20, 20), add_legend=True,
                           n_txt_per_cluster=3, alpha=1, s=0.5,
                           random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.
        """
        if labels is None:
            labels = self.labs

        return super(SingleLabelClassifiedSamples,
                     self).tsne_gradient_plot(
                        gradient=gradient, labels=labels,
                        selected_labels=selected_labels,
                        shuffle_label_colors=shuffle_label_colors,
                        title=title, xlab=xlab, ylab=ylab,
                        figsize=figsize,
                        add_legend=add_legend,
                        n_txt_per_cluster=n_txt_per_cluster,
                        alpha=alpha, s=s,
                        random_state=random_state,
                        **kwargs)

    def tsne_feature_gradient_plot(self, fid, transform=None, labels=None,
                                   selected_labels=None,
                                   shuffle_label_colors=False,
                                   title=None, xlab=None, ylab=None,
                                   figsize=(20, 20), add_legend=True,
                                   n_txt_per_cluster=3, alpha=1, s=0.5,
                                   random_state=None, **kwargs):
        """
        Plot the last t-SNE projection with the provided gradient as color.

        Parameters
        ----------
        fid: feature id scalar
            ID of the feature to be used for gradient plot.
        transform: callable
            Map transform on feature before plotting.
        """
        if labels is None:
            labels = self.labs

        return super(SingleLabelClassifiedSamples,
                     self).tsne_feature_gradient_plot(
                        fid=fid, transform=transform, labels=labels,
                        selected_labels=selected_labels,
                        shuffle_label_colors=shuffle_label_colors,
                        title=title, xlab=xlab, ylab=ylab,
                        figsize=figsize,
                        add_legend=add_legend,
                        n_txt_per_cluster=n_txt_per_cluster,
                        alpha=alpha, s=s,
                        random_state=random_state,
                        **kwargs)

    def feature_swarm_plot(self, fid, transform=None, labels=None,
                           selected_labels=None,
                           title=None, xlab=None, ylab=None,
                           figsize=(10, 10)):
        f_ind = self.f_id_to_ind([fid])[0]
        fx = self.f_ind_x_vec(f_ind)

        if transform is not None:
            if callable(transform):
                fx = np.array(list(map(transform, fx)))
            else:
                raise ValueError("transform must be a callable")

        if labels is not None and len(labels) != fx.shape[0]:
            raise ValueError("labels ({}) must have same length as "
                             "n_samples.".format(labels))
        else:
            labels = self.labs

        return swarm(fx, labels=labels, selected_labels=selected_labels,
                     title=title, xlab=xlab, ylab=ylab, figsize=figsize)

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
