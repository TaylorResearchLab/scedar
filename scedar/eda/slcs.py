import itertools

from collections import namedtuple

import numpy as np

from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split
import sklearn.utils

import matplotlib as mpl
import matplotlib.colors
import seaborn as sns

from collections import defaultdict

import xgboost as xgb

import inspect

from scedar.eda.sdm import SampleDistanceMatrix
from scedar.eda.plot import swarm
from scedar.eda.plot import heatmap
from scedar.eda import mdl
from scedar.eda import mtype
from scedar import utils


class SingleLabelClassifiedSamples(SampleDistanceMatrix):
    """Data structure of single label classified samples

    Attributes:
        _x (2D number array): (n_samples, n_features) data matrix.
        _d (2D number array): (n_samples, n_samples) distance matrix.
        _labs (list of labels): list of labels in the same type, int or str.
        _fids (list of feature IDs): list of feature IDs in the same type,
            int or str.
        _sids (list of sample IDs): list of sample IDs in the same type,
            int or str.
        _metric (str): Distance metric.

    Note:
        If sort by labels, the samples will be reordered, so that samples from
        left to right are from one label to another.
    """
    # sid, lab, fid, x

    def __init__(self, x, labs, sids=None, fids=None, d=None,
                 metric="cosine", use_pdist=True, nprocs=None):
        # sids: sample IDs. String or int.
        # labs: sample classified labels. String or int.
        # x: (n_samples, n_features)
        super(SingleLabelClassifiedSamples, self).__init__(
            x=x, d=d, metric=metric, use_pdist=use_pdist,
            sids=sids, fids=fids, nprocs=nprocs)

        mtype.check_is_valid_labs(labs)
        labs = np.array(labs)
        if self._sids.shape[0] != labs.shape[0]:
            raise ValueError("sids must have the same length as labs")
        self._labs = labs
        self._set_up_lab_rel_attrs()
        # keep a copy of original labels
        self._orig_labs = labs
        self._xgb_lut = {}
        return

    def _set_up_lab_rel_attrs(self):
        """Set up labels related attrs
        """
        self._uniq_labs, self._uniq_lab_cnts = np.unique(
            self._labs, return_counts=True)
        # {lab: array([sid0, ...]), ...}
        sid_lut = {}
        for ulab in self._uniq_labs:
            sid_lut[ulab] = self._sids[self._labs == ulab]
        self._sid_lut = sid_lut
        # {sid1: lab1, ...}
        lab_lut = {}
        # sids only contain unique values
        for i in range(self._sids.shape[0]):
            lab_lut[self._sids[i]] = self._labs[i]
        self._lab_lut = lab_lut

    def sort_by_labels(self):
        """
        Return a copy with sorted sample indices by labels and distances.
        """
        labels = np.array(self.labs)
        # slcs is empty
        if len(labels) == 0 or self._x.size == 0:
            return self.ind_x()
        uniq_labs = np.unique(labels)
        s_ind_lut = dict([(ulab, np.where(labels == ulab)[0])
                          for ulab in uniq_labs])
        # sort within each label
        for ulab in uniq_labs:
            # get sample indices of that class
            s_inds = s_ind_lut[ulab]
            # sort that class by distance to the first sample
            # get a list of distances to the frist sample
            s_dist_to_s0_list = [self._d[s_inds[0], s_inds[i]]
                                 for i in range(len(s_inds))]
            # sort indices by distances
            sorted_s_inds = s_inds[np.argsort(s_dist_to_s0_list,
                                              kind="mergesort")]
            # update lut
            s_ind_lut[ulab] = sorted_s_inds
        # sort classes by distances of first samples
        # frist sample indices
        lab_fs_inds = [s_ind_lut[ulab][0] for ulab in uniq_labs]
        # distance of first samples to the first class first sample
        lab_fs_dist_to_fc_list = [self._d[lab_fs_inds[0], lab_fs_inds[i]]
                                  for i in range(len(lab_fs_inds))]
        sorted_ulabs = uniq_labs[np.argsort(lab_fs_dist_to_fc_list,
                                            kind="mergesort")]
        sorted_s_inds = np.concatenate([s_ind_lut[ulab]
                                        for ulab in sorted_ulabs])
        return self.ind_x(sorted_s_inds)

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

        if self._use_pdist:
            return SingleLabelClassifiedSamples(
                x=self._x[selected_s_inds, :][:, selected_f_inds].copy(),
                d=self._d[selected_s_inds, :][:, selected_s_inds].copy(),
                labs=self._labs[selected_s_inds].tolist(),
                metric=self._metric,
                use_pdist=self._use_pdist,
                sids=self._sids[selected_s_inds].tolist(),
                fids=self._fids[selected_f_inds].tolist(),
                nprocs=self._nprocs)
        else:
            return SingleLabelClassifiedSamples(
                x=self._x[selected_s_inds, :][:, selected_f_inds].copy(),
                d=None,
                labs=self._labs[selected_s_inds].tolist(),
                metric=self._metric,
                use_pdist=self._use_pdist,
                sids=self._sids[selected_s_inds].tolist(),
                fids=self._fids[selected_f_inds].tolist(),
                nprocs=self._nprocs)

    def merge_labels(self, orig_labs_to_merge, new_lab):
        """Merge selected labels into a new label

        Args:
            orig_labs_to_merge (list of unique labels): original labels to be
                merged into a new label
            new_lab (label): new label of the merged labels

        Returns:
            None

        Note:
            Update labels in place.
        """
        if not mtype.is_valid_lab(new_lab):
            raise ValueError("new_lab, {}, must  be str or int")
        mtype.check_is_valid_labs(orig_labs_to_merge)
        # all labs must be unique
        if len(orig_labs_to_merge) != len(np.unique(orig_labs_to_merge)):
            raise ValueError("orig_labs_to_merge must all be unique")
        for ulab in orig_labs_to_merge:
            if ulab not in self._uniq_labs:
                raise ValueError("label {} not in original unique "
                                 "labels".format(ulab))
        updated_labs = self._labs.copy()
        for i, lab in enumerate(self._labs):
            if lab in orig_labs_to_merge:
                updated_labs[i] = new_lab
        self._labs = updated_labs
        self._set_up_lab_rel_attrs()
        return

    def relabel(self, labels):
        """
        Return a new SingleLabelClassifiedSamples with new labels.
        """
        return SingleLabelClassifiedSamples(
                   x=self._x.copy(), labs=labels, d=self._d.copy(),
                   sids=self._sids.tolist(),
                   fids=self._fids.tolist(),
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

    @staticmethod
    def select_labs_bool_inds(ref_labs, selected_labs):
        if selected_labs is None:
            return np.repeat(True, len(ref_labs))

        if np.isscalar(selected_labs):
            selected_labs = [selected_labs]

        ref_uniq_labs = np.unique(ref_labs).tolist()
        if not all([x in ref_uniq_labs for x in selected_labs]):
            raise ValueError("selected_labs: {} are not all existed "
                             "in the unique ref labels "
                             "{}".format(selected_labs, ref_uniq_labs))
        lab_selected_s_bool_inds = np.in1d(ref_labs, selected_labs)
        return lab_selected_s_bool_inds

    def lab_x_bool_inds(self, selected_labs):
        return self.select_labs_bool_inds(self._labs, selected_labs)

    def lab_x(self, selected_labs):
        lab_selected_s_bool_inds = self.lab_x_bool_inds(selected_labs)
        return self.ind_x(lab_selected_s_bool_inds)

    @staticmethod
    def _xgb_train_runner(x, lab_inds, fids, test_size=0.3,
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
        # xgb only takes string as feature names
        str_fids = list(map(str, fids))
        orig_fid_lut = dict(zip(str_fids, fids))
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
        eval_stats = [[(eval_name + " " + mname,
                        mval_list[num_boost_round-1])
                       for mname, mval_list in eval_dict.items()]
                      for eval_name, eval_dict in evals_result.items()]
        # {feature_name: fscore, ...}
        fscore_dict = bst.get_fscore()
        orig_fid_fscore_list = [(orig_fid_lut[istr_fid], ifscore)
                                for istr_fid, ifscore in fscore_dict.items()]
        sorted_fscore_list = sorted(orig_fid_fscore_list, key=lambda t: t[1],
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
            [(feature_id, mean of fscore across all bootstrapping rounds,
            standard deviation of fscore across all bootstrapping rounds,
            number of times used all bootstrapping rounds), ...]
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
        fids = lab_selected_slcs.fids
        if shuffle_features:
            feature_inds = np.arange(lab_selected_slcs._x.shape[1])
            feature_inds, fids = sklearn.utils.shuffle(
                feature_inds, fids)
        else:
            feature_inds = slice(None, None)
        # perform bootstrapping if necessary
        num_bootstrap_round = int(num_bootstrap_round)
        if num_bootstrap_round <= 0:
            # no bootstrapping
            # _xgb_train_runner returns (fscores, bst, eval_stats)
            fscores, bst, eval_stats = self._xgb_train_runner(
                lab_selected_slcs._x[:, feature_inds],
                lab_inds, fids, test_size=test_size,
                num_boost_round=num_boost_round,
                xgb_params=xgb_params, random_state=random_state,
                nprocs=nprocs, silent=silent)
            # make sorted_fs_list consistent to the bootstrapped one
            fs_list = [(t[0], t[1], 0, 1, [t[1]])
                       for t in fscores]
            sorted_fs_list = sorted(fs_list, key=lambda t: (t[3], t[1]),
                                    reverse=True)
            print(eval_stats)
            bst_list = [bst]
        else:
            # do bootstrapping
            # ([dict of scores], [list of bsts], dict of eval stats)
            fs_dict = defaultdict(list)
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
                    bs_lab_inds, fids, test_size=test_size,
                    num_boost_round=num_boost_round,
                    xgb_params=xgb_params, random_state=random_state,
                    nprocs=nprocs, silent=silent)
                # Sum fscores
                for fid, fs in fscores:
                    fs_dict[fid] += [fs]
                bst_list.append(bst)
                # est: eval stats tuple
                # [ [('train...', float), ...],
                #   [('test...', float), ...] ]
                for elist in eval_stats:
                    for ename, evalue in elist:
                        eval_stats_dict[ename].append(evalue)
                if shuffle_features:
                    feature_inds, fids = sklearn.utils.shuffle(
                        feature_inds, fids)
            # score summary: average, std, times showed up
            fid_s_list = []
            for fid, fs in fs_dict.items():
                fid_s_list.append((fid, np.mean(fs), np.std(fs, ddof=0),
                                   len(fs), fs))
            sorted_fs_list = sorted(fid_s_list, key=lambda t: (t[3], t[1]),
                                    reverse=True)
            # calculate mean +/- std of eval stats
            for ename, evalue_list in eval_stats_dict.items():
                print("{}: mean {}, std {}".format(
                    ename, np.mean(evalue_list), np.std(evalue_list, ddof=1)))
        # return same things for two branches
        return sorted_fs_list, bst_list

    def feature_importance_distintuishing_labs(self, selected_labs,
                                               test_size=0.3,
                                               num_boost_round=10, nprocs=1,
                                               random_state=None, silent=1,
                                               xgb_params=None,
                                               num_bootstrap_round=0,
                                               bootstrap_size=None,
                                               shuffle_features=False):
        """
        Use xgboost to compare selected labels and others.
        """
        selected_s_bool_inds = self.lab_x_bool_inds(selected_labs)
        # binary labs distinguishing selected and non-selected
        io_bin_lab_arr = ["selected" if s else "non-selected"
                          for s in selected_s_bool_inds]
        # create a new SLCS instance with new labels
        nl_slcs = self.relabel(io_bin_lab_arr)
        fi_res = nl_slcs.feature_importance_across_labs(
            ["selected", "non-selected"], test_size=test_size,
            num_boost_round=num_boost_round, nprocs=nprocs,
            random_state=random_state, silent=silent,
            xgb_params=xgb_params, num_bootstrap_round=num_bootstrap_round,
            bootstrap_size=bootstrap_size, shuffle_features=shuffle_features)
        return fi_res

    def feature_importance_each_lab(self, test_size=0.3, num_boost_round=10,
                                    nprocs=1, random_state=None, silent=1,
                                    xgb_params=None, num_bootstrap_round=0,
                                    bootstrap_size=None,
                                    shuffle_features=False):
        """
        Use xgboost to compare each label with others. Experimental.
        """
        # Construct important feature lut
        # {ulab0: [if1, if2, ...], ...}
        ulab_fi_lut = defaultdict(list)
        for ulab in self._uniq_labs:
            # get bool indices of current label
            ulab_s_bool_inds = self.lab_x_bool_inds(ulab)
            # compare current label with other samples
            fi_res = self.feature_importance_distintuishing_labs(
                ulab, test_size=test_size,
                num_boost_round=num_boost_round, nprocs=nprocs,
                random_state=random_state, silent=silent,
                xgb_params=xgb_params,
                num_bootstrap_round=num_bootstrap_round,
                bootstrap_size=bootstrap_size,
                shuffle_features=shuffle_features)

            for fid in [t[0] for t in fi_res[0]]:
                fx = self.f_id_x_vec(fid)
                # current label values
                ulab_x = fx[ulab_s_bool_inds]
                # other values
                other_x = fx[np.logical_not(ulab_s_bool_inds)]
                # current lab mean
                ulab_x_mean = np.mean(ulab_x)
                # other mean
                other_x_mean = np.mean(other_x)
                # mean fold change
                ulab_mfc = (ulab_x_mean - other_x_mean) / ulab_x_mean
                # ks test result
                ks_res = ks_2samp(ulab_x, other_x)
                ulab_fi_lut[ulab].append((fid, ulab_mfc, ks_res.pvalue))
            ulab_fi_lut[ulab].sort(key=lambda t: t[1], reverse=True)
            ulab_fi_lut[ulab] = [t for t in ulab_fi_lut[ulab]]
        return ulab_fi_lut

    def tsne_plot(self, gradient=None, labels=None,
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
                     self).tsne_plot(
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

    def dmat_heatmap(self, selected_labels=None, col_labels=None,
                     transform=None,
                     title=None, xlab=None, ylab=None, figsize=(10, 10),
                     **kwargs):
        """
        Plot distance matrix with rows colored by current labels.
        """
        selected_s_bool_inds = self.lab_x_bool_inds(selected_labels)
        selected_labels = self._labs[selected_s_bool_inds].tolist()
        selected_d = self._d[selected_s_bool_inds, :][:, selected_s_bool_inds]
        return heatmap(selected_d, row_labels=selected_labels,
                       col_labels=col_labels, transform=transform,
                       title=title, xlab=xlab, ylab=ylab,
                       figsize=figsize, **kwargs)

    def xmat_heatmap(self, selected_labels=None, selected_fids=None,
                     col_labels=None, transform=None,
                     title=None, xlab=None, ylab=None, figsize=(10, 10),
                     **kwargs):
        """
        Plot x as heatmap.
        """
        selected_s_bool_inds = self.lab_x_bool_inds(selected_labels)
        selected_s_ids = self._sids[selected_s_bool_inds]
        selected_slcs = self.id_x(selected_s_ids, selected_fids)
        return heatmap(selected_slcs._x, row_labels=selected_slcs.labs,
                       col_labels=col_labels, transform=transform,
                       title=title, xlab=xlab, ylab=ylab,
                       figsize=figsize, **kwargs)

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
            min_sid_sorted_sep_lab_ind_list = [
                sep_lab_min_sid_list.index(x)
                for x in sorted_sep_lab_min_sid_list
            ]
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
        np.testing.assert_array_equal(
            lab_sorted_lab_arr[np.argsort(lab_sorted_sid_arr)],
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


class MDLSingleLabelClassifiedSamples(SingleLabelClassifiedSamples):
    """
    MDLSingleLabelClassifiedSamples inherits SingleLabelClassifiedSamples to
    offer MDL operations.

    Args:
        x (2d number array): data matrix
        labs (list of str or int): labels
        sids (list of str or int): sample ids
        fids (list of str or int): feature ids
        encode_type ("auto", "data", or "distance"): Type of values to encode.
            If "auto", encode data when n_features <= 100.
        mdl_method (mdl.Mdl): If None, use ZeroIGKdeMdl for encoded values
            with >= 50% zeros, and use GKdeMdl otherwise.
        d (2d number array): distance matrix
        metric (str): distance metric for scipy
        nprocs (int)

    Attributes:
        _mdl_method (.mdl.Mdl)
    """
    LabMdlResult = namedtuple("LabMdlResult",
                              ["ulab_mdl_sum", "ulab_s_inds", "ulab_cnts",
                               "ulab_mdls", "cluster_mdl"])

    def __init__(self, x, labs, sids=None, fids=None,
                 encode_type="data", mdl_method=mdl.ZeroIGKdeMdl,
                 d=None, metric="correlation", nprocs=None):
        super(MDLSingleLabelClassifiedSamples, self).__init__(
            x=x, labs=labs, sids=sids, fids=fids, d=d, metric=metric,
            nprocs=nprocs)
        # initialize encode type
        if encode_type not in ("auto", "data", "distance"):
            raise ValueError("encode_type must in "
                             "('auto', 'data', 'distance')."
                             "Provided: {}".format(encode_type))
        if encode_type == "auto":
            if self._x.shape[1] > 100:
                encode_type = "distance"
            else:
                encode_type = "data"
        self._encode_type = encode_type
        # initialize mdl method
        if mdl_method is None:
            if encode_type == "data":
                ex = self._x
            else:
                ex = self._d
            if ex.size == 0:
                # empty matrix
                mdl_method = mdl.GKdeMdl
            else:
                n_nonzero = np.count_nonzero(ex)
                if n_nonzero / ex.size > 0.5:
                    mdl_method = mdl.GKdeMdl
                else:
                    mdl_method = mdl.ZeroIGKdeMdl
        self._mdl_method = mdl_method

    @staticmethod
    def per_col_encoders(x, encode_type, mdl_method=mdl.ZeroIGKdeMdl, nprocs=1,
                         verbose=False):
        """Compute mdl encoder for each column

        Args:
            x (2d number array)
            encode_type ("data" or "distance")
            mdl_method (mdl.Mdl)
            nprocs (int)
            verbose (bool)

        Returns:
            :obj: list of column mdl encoders of x
        """
        # verbose is not implemented
        if not inspect.isclass(mdl_method):
            raise ValueError("method must be a subclass of eda.mdl.Mdl")

        if not issubclass(mdl_method, mdl.Mdl):
            raise ValueError("method must be a subclass of eda.mdl.Mdl")

        if x.ndim != 2:
            raise ValueError("x should be 2D. x.shape: {}".format(x.shape))

        nprocs = max(int(nprocs), 1)
        if encode_type == "data":
            col_encoders = utils.parmap(
                lambda x1d: mdl_method(x1d), x.T, nprocs)
        elif encode_type == "distance":
            # distance
            s_inds = list(range(x.shape[0]))
            xs_for_map = [x[s_inds[i], s_inds[:i] + s_inds[i+1:]]
                          for i in s_inds]

            def single_s_mdl_encoder(x1d):
                # copy indices for parallel processing
                return mdl_method(x1d)
            col_encoders = utils.parmap(
                single_s_mdl_encoder, xs_for_map, nprocs)
        else:
            raise NotImplementedError("unknown encode_type: "
                                      "{}".format(encode_type))

        return col_encoders

    def no_lab_mdl(self, nprocs=1, verbose=False):
        """Compute mdl of each feature without separating samples by labels

        Args:
            nprocs (int)
            verbose (bool): Not implemented

        Returns:
            float: mdl of matrix without separating samples by labels
        """
        # TODO: implement verbose
        if self._encode_type == "data":
            col_encoders = self.per_col_encoders(self._x, self._encode_type,
                                                 self._mdl_method, nprocs,
                                                 verbose)
            col_mdl_sum = np.sum(list(map(lambda e: e.mdl, col_encoders)))
        elif self._encode_type == "distance":
            col_encoders = self.per_col_encoders(self._d, self._encode_type,
                                                 self._mdl_method, nprocs,
                                                 verbose)
            col_mdl_sum = 0
            ulab_s_ind_list = [np.where(self._labs == ulab)[0].tolist()
                               for ulab in self._uniq_labs]
            for s_inds in ulab_s_ind_list:
                n_s_inds = len(s_inds)
                rn_inds = list(range(n_s_inds))
                # [(encoder, x), ...]
                enc_x_tups = []
                for i in rn_inds:
                    i_s_ind = s_inds[i]
                    non_i_s_inds = s_inds[:i] + s_inds[i+1:]
                    enc_x_tups.append((col_encoders[i_s_ind],
                                       self._d[non_i_s_inds, i_s_ind]))
                mdls = utils.parmap(lambda ext: ext[0].encode(ext[1]),
                                    enc_x_tups, nprocs)
                col_mdl_sum += sum(mdls)
        else:
            raise NotImplementedError("Do not change encode_type after init. "
                                      "Unknown encode type "
                                      "{}".format(self._encode_type))
        return col_mdl_sum

    @staticmethod
    def compute_cluster_mdl(labs, cl_mdl_scale_factor=1):
        """Additional MDL for encoding the cluster

        - labels are encoded by multinomial distribution
        - parameters are encoded by 32bit float
          np.log(2**32) = 22.18070977791825
        - scaled by factor

        TODO: formalize parameter mdl
        """
        uniq_labs, uniq_lab_cnts = np.unique(labs, return_counts=True)
        n_uniq_labs = len(uniq_lab_cnts)
        # make a flat list of labels
        int_labs = list(itertools.chain.from_iterable(
            [[i]*uniq_lab_cnts[i] for i in range(n_uniq_labs)]))
        mn_mdl = mdl.MultinomialMdl(int_labs).mdl
        param_mdl = 22.18070977791825 * n_uniq_labs
        return (mn_mdl + param_mdl) * cl_mdl_scale_factor

    def lab_mdl(self, cl_mdl_scale_factor=1, nprocs=1, verbose=False,
                ret_internal=False):
        """Compute mdl of each feature after separating samples by labels

        Args:
            cl_mdl_scale_factor (float): multiplies cluster related mdl by this
                number
            nprocs (int)
            verbose (bool): Not implemented

        Returns:
            float: mdl of matrix after separating sampels by labels
        """
        # compute cluster label overhead mdl
        cluster_mdl = self.compute_cluster_mdl(
            self._labs, cl_mdl_scale_factor=cl_mdl_scale_factor)
        # compute mdl for data points
        n_uniq_labs = self._uniq_labs.shape[0]
        ulab_s_ind_list = [np.where(self._labs == ulab)[0].tolist()
                           for ulab in self._uniq_labs]
        # summarize mdls of all clusters
        if self._encode_type == "distance":
            ulab_mdls = [MDLSingleLabelClassifiedSamples(
                            self._x[s_inds], labs=[0]*len(s_inds),
                            encode_type=self._encode_type,
                            mdl_method=self._mdl_method,
                            d=self._d[s_inds][:, s_inds],
                            metric=self._metric,
                            nprocs=self._nprocs).no_lab_mdl()
                         for s_inds in ulab_s_ind_list]
        elif self._encode_type == "data":
            # do not pass d
            ulab_mdls = [MDLSingleLabelClassifiedSamples(
                            self._x[s_inds], labs=[0]*len(s_inds),
                            encode_type=self._encode_type,
                            mdl_method=self._mdl_method,
                            metric=self._metric,
                            nprocs=self._nprocs).no_lab_mdl()
                         for s_inds in ulab_s_ind_list]
        else:
            raise NotImplementedError("Do not change encode_type after init. "
                                      "Unknown encode type "
                                      "{}".format(self._encode_type))
        # add cluster overhead mdl to each cluster
        ulab_cnt_ratios = self._uniq_lab_cnts / np.int_(self._x.shape[0])
        ulab_cl_mdls = [ulab_mdls[i] + cluster_mdl * ulab_cnt_ratios[i]
                        for i in range(n_uniq_labs)]
        ulab_mdl_sum = np.sum(ulab_cl_mdls)
        lab_mdl_res = self.LabMdlResult(ulab_mdl_sum, ulab_s_ind_list,
                                        self._uniq_lab_cnts.tolist(),
                                        ulab_cl_mdls, cluster_mdl)
        if ret_internal:
            return lab_mdl_res, ulab_mdls
        else:
            return lab_mdl_res

    def encode(self, qx, col_summary_func=sum,
               non_zero_only=False, nprocs=1, verbose=False):
        """Encode input array qx with fitted code without label

        Args:
            qx (2d np number array)
            col_summary_func (callable): function applied on column mdls
            non_zero_only (bool): whether to encode non-zero entries only
            nprocs (int)
            verbose (bool)

        Returns:
            float: mdl for encoding qx
        """
        if not callable(col_summary_func):
            raise ValueError("col_summary_func must be callable")

        if self._encode_type == "data":
            ex = self._x
        elif self._encode_type == "distance":
            ex = self._d
        else:
            raise NotImplementedError("Do not change encode_type after init. "
                                      "Unknown encode type "
                                      "{}".format(self._encode_type))

        if qx.ndim != 2 or qx.shape[1] != ex.shape[1]:
            raise ValueError("Array to encode should have the same number of"
                             "columns as the encoded x")

        col_encoders = self.per_col_encoders(
            ex, self._encode_type, self._mdl_method, nprocs, verbose=verbose)

        ncols = ex.shape[1]
        q_x_cols = []
        for i in range(ncols):
            x_col = qx[:, i]
            # mdl_method is valid after running per_col_encoders
            if non_zero_only:
                q_x_cols.append(x_col[np.nonzero(x_col)])
            else:
                # all mdl methods here are valid
                # this branch is GKdeMdl
                q_x_cols.append(x_col)
        # (mdl, qx) tuple
        mdl_qxcol_tups = list(zip(col_encoders, q_x_cols))

        encode_q_col_mdls = utils.parmap(
            lambda ext: ext[0].encode(ext[1]), mdl_qxcol_tups, nprocs=nprocs)

        encode_x_mdl = col_summary_func(encode_q_col_mdls)
        return encode_x_mdl
