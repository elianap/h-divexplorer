import pandas as pd
import numpy as np
from mlxtend.frequent_patterns.apriori import (
    generate_new_combinations_low_memory,
    generate_new_combinations,
)
from mlxtend.frequent_patterns import fpcommon as fpc


# from .import_datasets import *
# from .utils_metrics_FPx import *

# from utils_significance import *
map_beta_distribution = {
    "d_fpr": {"T": ["fp"], "F": ["tn"]},
    "d_fnr": {"T": ["fn"], "F": ["tp"]},
    "d_accuracy": {"T": ["tp", "tn"], "F": ["fp", "fn"]},
    "d_fpr_abs": {"T": ["fp"], "F": ["tn"]},
    "d_fnr_abs": {"T": ["fn"], "F": ["tp"]},
    "d_accuracy_abs": {"T": ["tp", "tn"], "F": ["fp", "fn"]},
    "d_posr": {"T": ["tp", "fn"], "F": ["tn", "fp"]},
    "d_negr": {"T": ["tn", "fp"], "F": ["tp", "fn"]},
    "d_error": {"T": ["fp", "fn"], "F": ["tp", "tn"]},
    "d_ppv": {"T": ["tp"], "F": ["fp"]},
    "d_tpr": {"T": ["tp"], "F": ["fn"]},
    "d_tnr": {"T": ["tn"], "F": ["fp"]},
    "d_npv": {"T": ["tn"], "F": ["fn"]},
    "d_fdr": {"T": ["fp"], "F": ["tp"]},
    "d_for": {"T": ["fn"], "F": ["tn"]},
}

VIZ_COL_NAME = "viz"


def oneHotEncoding(dfI):
    attributes = dfI.columns
    X_one_hot = dfI.copy()
    X_one_hot = pd.get_dummies(X_one_hot, prefix_sep="=", columns=attributes)
    X_one_hot.reset_index(drop=True, inplace=True)
    return X_one_hot


def _compute_t_test(df, col_mean, col_var, mean_d, var_d):
    return (abs(df[col_mean] - mean_d)) / ((df[col_var] + var_d) ** 0.5)


# def _compute_std_beta_distribution(FPb):
#    return ((FPb.a*FPb.b)/((FPb.a+FPb.b)**2*(FPb.a+FPb.b+1)))**(1/2)


def _compute_variance_beta_distribution(FPb):
    return (FPb.a * FPb.b) / ((FPb.a + FPb.b) ** 2 * (FPb.a + FPb.b + 1))


def _compute_mean_beta_distribution(FPb):
    return FPb.a / (FPb.a + FPb.b)


def _keep_only_admitted(X_oh, incompatible, keep_items):
    """
    Args
        X_oh (pandas.DataFrame): dataframe in one hot encoding. Discretized items + generalized ones
        incompatible (dict): {attribute: {frozenset of incompatible items}}
        keep_items (dict): {attribute: [admitted items]}

    Returs:
        pandas.DataFrame: updated dataframe in one hot encoding with only the admitted items
        dict: updated incompatible dictionary
    """
    # TODO remove deepcopy
    from copy import deepcopy

    X_pruned = deepcopy(X_oh)
    incompatible_pruned = deepcopy(incompatible)

    oh_cols = list(X_pruned.columns)

    items_removed = 0
    for e, col in enumerate(oh_cols):
        s = col.split("=")
        attr = s[0]
        val = "=".join(s[1:])

        if attr in keep_items:
            if val not in keep_items[attr]:

                id_item_to_remove = e - items_removed
                items_removed += 1
                X_pruned.drop(columns=[col], inplace=True)
                for attribute in incompatible_pruned:
                    vals = []
                    for val in incompatible_pruned[attribute]:
                        if val == id_item_to_remove:
                            continue
                        if val > id_item_to_remove:
                            val -= 1
                        vals.append(val)
                    incompatible_pruned[attribute] = frozenset(vals)
    return X_pruned, incompatible_pruned


def _keep_only_admitted_single_discretization(X_oh, keep_items):
    """
    We directly drop the not admitted items
    Args
        X_oh (pandas.DataFrame): dataframe in one hot encoding. Discretized items + generalized ones

        keep_items (dict): {attribute: [admitted items]}

    Returs:
        pandas.DataFrame: updated dataframe in one hot encoding with only the admitted items
        dict: updated incompatible dictionary
    """
    # TODO remove deepcopy
    from copy import deepcopy

    X_pruned = deepcopy(X_oh)

    oh_cols = list(X_pruned.columns)

    items_removed = 0
    for e, col in enumerate(oh_cols):
        s = col.split("=")
        attr = s[0]
        val = "=".join(s[1:])

        if attr in keep_items:
            if val not in keep_items[attr]:
                X_pruned.drop(columns=[col], inplace=True)
    return X_pruned


class FP_DivergenceExplorer:
    def __init__(
        self,
        X_discrete,
        true_class_name,
        predicted_class_name=None,
        class_map={},
        ignore_cols=[],
        log_loss_values=None,
        clf=None,
        dataset_name="",
        type_cl="",
        generalizations_obj=None,
        preserve_interval=None,
        sep="=",
        already_in_one_hot_encoding=False,
        keep_only_positive_divergent_items=None,
    ):
        # TODO now function in import dataset

        if keep_only_positive_divergent_items is not None:
            if type(keep_only_positive_divergent_items) != dict:
                raise ValueError(
                    "The input should be a dictionary. Attribute: [admitted items]"
                )

        cols = (
            [true_class_name, predicted_class_name] + ignore_cols
            if predicted_class_name is not None
            else [true_class_name] + ignore_cols
        )

        ### Handling generalization/taxonomy
        self.generalizations_obj = generalizations_obj
        self.hasGeneralization = False if generalizations_obj is None else True
        if already_in_one_hot_encoding:
            self.X = X_discrete.drop(columns=cols)
        else:
            self.X = oneHotEncoding(X_discrete.drop(columns=cols))
        self.y = X_discrete[[true_class_name]].copy()
        self.drop = None
        self.incompatible_items = None

        if (self.hasGeneralization) and (preserve_interval):
            drop = {}
            for attribute_pi in preserve_interval:
                items_keep = preserve_interval[attribute_pi]
                col_values = [
                    sep.join(i.split(sep)[1:])
                    for i in self.X.columns
                    if attribute_pi in i
                ]
                for e, col_val in enumerate(col_values):
                    if col_val not in items_keep:
                        if attribute_pi not in drop:
                            drop[attribute_pi] = []
                        drop[attribute_pi].append((col_val, e))
            self.drop = drop

        ### Handling generalization/taxonomy
        if self.generalizations_obj is not None:
            (
                self.X,
                self.new_remapped_columns,
            ) = self.generalizations_obj.extend_dataset_with_hierarchy(
                self.X, drop_intervals=self.drop, sep=sep
            )

        if self.hasGeneralization:
            self.incompatible_items = (
                self.generalizations_obj.get_attributes_incompatibility()
            )

            if self.new_remapped_columns:

                for attribute, incompatible_set in self.incompatible_items.items():

                    replaced_incompatible = frozenset(
                        [
                            self.new_remapped_columns[i]
                            for i in incompatible_set
                            if i in self.new_remapped_columns
                        ]
                    )
                    self.incompatible_items[attribute] = replaced_incompatible

            if keep_only_positive_divergent_items is not None:
                self.X, self.incompatible_items = _keep_only_admitted(
                    self.X, self.incompatible_items, keep_only_positive_divergent_items
                )

        else:

            if keep_only_positive_divergent_items is not None:
                # Keep only the admitted ones also in the case of single discretizatiion without generalization
                self.X = _keep_only_admitted_single_discretization(
                    self.X, keep_only_positive_divergent_items
                )

        self.y_predicted = (
            X_discrete[predicted_class_name].copy().values
            if predicted_class_name is not None
            else X_discrete[true_class_name].copy().values
        )

        self.log_loss_values = log_loss_values

        self.dataset_name = dataset_name
        self.clf = clf
        self.dataset_name = dataset_name
        self.type_cl = type_cl

        self.FP_metric_support = {}

        self.y_true_pred = None
        self.y_true_pred = self.y.copy()
        self.y_true_pred.columns = ["true_class"]  # TODO class1?
        self.y_true_pred = self.y_true_pred.assign(predicted=self.y_predicted)
        # self.y_true_pred=self.y_true_pred.rename(columns={"class": "true_class"})

        self.class_map = class_map
        if self.class_map == {}:
            from sklearn.utils.multiclass import unique_labels

            labels = np.sort(unique_labels(self.y, self.y_predicted))
            if len(labels) > 2:
                # todo error
                print("Binary class")
                raise ValueError(f"Not binary problem:{len(labels)}")
            self.class_map = {"N": labels[0], "P": labels[1]}

    def instanceConfusionMatrix(self, df):
        # TODO
        df["tn"] = (
            (df.true_class == df.predicted) & (df.true_class == self.class_map["N"])
        ).astype(int)
        df["fp"] = (
            (df.true_class != df.predicted) & (df.true_class == self.class_map["N"])
        ).astype(int)
        df["tp"] = (
            (df.true_class == df.predicted) & (df.true_class == self.class_map["P"])
        ).astype(int)
        df["fn"] = (
            (df.true_class != df.predicted) & (df.true_class == self.class_map["P"])
        ).astype(int)
        return df

    def apriori_divergence(
        self,
        df,
        df_true_pred,
        min_support=0.5,
        use_colnames=False,
        max_len=None,
        verbose=0,
        low_memory=False,
        cols_orderTP=["tn", "fp", "fn", "tp"],
        sortedV="support",
    ):
        """

        Returns
        -----------
        pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
          that are >= `min_support` and < than `max_len`
          (if `max_len` is not None).
          Each itemset in the 'itemsets' column is of type `frozenset`,
          which is a Python built-in type that behaves similarly to
          sets except that it is immutable
          (For more info, see
          https://docs.python.org/3.6/library/stdtypes.html#frozenset).
        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
        """

        def filterColumns(df_filter, cols):
            return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]

        def sum_values(_x):
            out = np.sum(_x, axis=0)
            return np.array(out).reshape(-1)

        def _support(_x, _n_rows, _is_sparse):
            """DRY private method to calculate support as the
            row-wise sum of values / number of rows
            Parameters
            -----------
            _x : matrix of bools or binary
            _n_rows : numeric, number of rows in _x
            _is_sparse : bool True if _x is sparse
            Returns
            -----------
            np.array, shape = (n_rows, )
            Examples
            -----------
            For usage examples, please see
            http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
            """
            out = np.sum(_x, axis=0) / _n_rows
            return np.array(out).reshape(-1)

        if min_support <= 0.0:
            raise ValueError(
                "`min_support` must be a positive "
                "number within the interval `(0, 1]`. "
                "Got %s." % min_support
            )

        fpc.valid_input_check(df)

        if hasattr(df, "sparse"):
            # DataFrame with SparseArray (pandas >= 0.24)
            if df.size == 0:
                X = df.values
            else:
                X = df.sparse.to_coo().tocsc()
            is_sparse = True
        else:
            # dense DataFrame
            X = df.values
            is_sparse = False
        support = _support(X, X.shape[0], is_sparse)
        ary_col_idx = np.arange(X.shape[1])
        support_dict = {0: 1, 1: support[support >= min_support]}
        itemset_dict = {0: [()], 1: ary_col_idx[support >= min_support].reshape(-1, 1)}
        conf_metrics = {
            0: np.asarray([sum_values(df_true_pred[cols_orderTP])]),
            1: np.asarray(
                [
                    sum_values(filterColumns(df_true_pred, item)[cols_orderTP])
                    for item in itemset_dict[1]
                ]
            ),
        }
        max_itemset = 1
        rows_count = float(X.shape[0])

        all_ones = np.ones((int(rows_count), 1))

        while max_itemset and max_itemset < (max_len or float("inf")):
            next_max_itemset = max_itemset + 1

            # With exceptionally large datasets, the matrix operations can use a
            # substantial amount of memory. For low memory applications or large
            # datasets, set `low_memory=True` to use a slower but more memory-
            # efficient implementation.
            if low_memory:
                combin = generate_new_combinations_low_memory(
                    itemset_dict[max_itemset], X, min_support, is_sparse
                )
                # slightly faster than creating an array from a list of tuples
                combin = np.fromiter(combin, dtype=int)
                combin = combin.reshape(-1, next_max_itemset + 1)

                if combin.size == 0:
                    break
                if verbose:
                    print(
                        "\rProcessing %d combinations | Sampling itemset size %d"
                        % (combin.size, next_max_itemset),
                        end="",
                    )

                itemset_dict[next_max_itemset] = combin[:, 1:]
                support_dict[next_max_itemset] = combin[:, 0].astype(float) / rows_count
                max_itemset = next_max_itemset
                # TODO
            else:
                combin = generate_new_combinations(itemset_dict[max_itemset])
                combin = np.fromiter(combin, dtype=int)
                combin = combin.reshape(-1, next_max_itemset)

                if combin.size == 0:
                    break
                if verbose:
                    print(
                        "\rProcessing %d combinations | Sampling itemset size %d"
                        % (combin.size, next_max_itemset),
                        end="",
                    )

                if is_sparse:
                    _bools = X[:, combin[:, 0]] == all_ones
                    for n in range(1, combin.shape[1]):
                        _bools = _bools & (X[:, combin[:, n]] == all_ones)
                else:
                    _bools = np.all(X[:, combin], axis=2)
                support = _support(np.array(_bools), rows_count, is_sparse)
                _mask = (support >= min_support).reshape(-1)
                if any(_mask):
                    itemset_dict[next_max_itemset] = np.array(combin[_mask])
                    support_dict[next_max_itemset] = np.array(support[_mask])
                    conf_metrics[next_max_itemset] = np.asarray(
                        [
                            sum_values(
                                filterColumns(df_true_pred, itemset)[cols_orderTP]
                            )
                            for itemset in itemset_dict[next_max_itemset]
                        ]
                    )
                    max_itemset = next_max_itemset
                else:
                    # Exit condition
                    break

        all_res = []
        for k in sorted(itemset_dict):
            support = pd.Series(support_dict[k])
            itemsets = pd.Series(
                [frozenset(i) for i in itemset_dict[k]], dtype="object"
            )
            # conf_matrix_col=pd.Series(list(conf_metrics[k]))
            conf_metrics_cols = pd.DataFrame(
                list(conf_metrics[k]), columns=cols_orderTP
            )

            res = pd.concat((support, itemsets, conf_metrics_cols), axis=1)
            all_res.append(res)

        res_df = pd.concat(all_res)
        res_df.columns = ["support", "itemsets"] + cols_orderTP

        if use_colnames:
            mapping = {idx: item for idx, item in enumerate(df.columns)}
            res_df["itemsets"] = res_df["itemsets"].apply(
                lambda x: frozenset([mapping[i] for i in x])
            )

        res_df["length"] = res_df["itemsets"].str.len()
        res_df["support_count"] = np.sum(res_df[cols_orderTP], axis=1)

        res_df.sort_values(sortedV, ascending=False, inplace=True)
        res_df = res_df.reset_index(drop=True)

        if verbose:
            print()  # adds newline if verbose counter was used

        return res_df

    def computeDivergenceItemsets(
        self,
        fm_df,
        metrics=["d_fpr", "d_fnr", "d_accuracy"],
        cols_orderTP=["tn", "fp", "fn", "tp"],
    ):

        # TODO - REFACTOR CODE

        if "d_fpr" in metrics:
            from .utils_metrics_FPx import fpr_df

            fm_df["fpr"] = fpr_df(fm_df[cols_orderTP])

        if "d_fnr" in metrics:
            from .utils_metrics_FPx import fnr_df

            fm_df["fnr"] = fnr_df(fm_df[cols_orderTP])

        if "d_accuracy" in metrics:
            from .utils_metrics_FPx import accuracy_df

            fm_df["accuracy"] = accuracy_df(fm_df[cols_orderTP])

        if "d_error" in metrics:
            from .utils_metrics_FPx import classification_error_df

            fm_df["error"] = classification_error_df(fm_df[cols_orderTP])

        if "d_ppv" in metrics:
            from .utils_metrics_FPx import positive_predicted_value_df

            fm_df["ppv"] = positive_predicted_value_df(fm_df[cols_orderTP])

        if "d_tpr" in metrics:
            from .utils_metrics_FPx import true_positive_rate_df

            fm_df["tpr"] = true_positive_rate_df(fm_df[cols_orderTP])

        if "d_tnr" in metrics:
            from .utils_metrics_FPx import true_negative_rate_df

            fm_df["tnr"] = true_negative_rate_df(fm_df[cols_orderTP])

        if "d_npv" in metrics:
            from .utils_metrics_FPx import negative_predicted_value_df

            fm_df["npv"] = negative_predicted_value_df(fm_df[cols_orderTP])

        if "d_fdr" in metrics:
            from .utils_metrics_FPx import false_discovery_rate_df

            fm_df["fdr"] = false_discovery_rate_df(fm_df[cols_orderTP])

        if "d_for" in metrics:
            from .utils_metrics_FPx import false_omission_rate_df

            fm_df["for"] = false_omission_rate_df(fm_df[cols_orderTP])

        if "d_posr" in metrics:
            # TODO
            from .utils_metrics_FPx import getInfoRoot

            rootIndex = getInfoRoot(fm_df).index
            from .utils_metrics_FPx import get_pos, posr_df

            fm_df["P"] = get_pos(fm_df[cols_orderTP])
            fm_df["posr"] = posr_df(fm_df[cols_orderTP])
            fm_df["d_posr"] = fm_df["posr"] - fm_df.loc[rootIndex]["posr"].values[0]
        if "d_negr" in metrics:
            # TODO
            from .utils_metrics_FPx import getInfoRoot

            rootIndex = getInfoRoot(fm_df).index
            from .utils_metrics_FPx import get_neg, negr_df

            fm_df["N"] = get_neg(fm_df[cols_orderTP])
            fm_df["negr"] = negr_df(fm_df[cols_orderTP])
            fm_df["d_negr"] = fm_df["negr"] - fm_df.loc[rootIndex]["negr"].values[0]

        from .utils_metrics_FPx import getInfoRoot

        infoRoot = getInfoRoot(fm_df)

        if "d_fnr" in metrics:
            fm_df["d_fnr"] = fm_df["fnr"] - infoRoot["fnr"].values[0]
        if "d_fpr" in metrics:
            fm_df["d_fpr"] = fm_df["fpr"] - infoRoot["fpr"].values[0]
        if "d_accuracy" in metrics:
            fm_df["d_accuracy"] = fm_df["accuracy"] - infoRoot["accuracy"].values[0]

        # Classification error
        if "d_error" in metrics:
            fm_df["d_error"] = fm_df["error"] - infoRoot["error"].values[0]

        # Precision or positive predictive value (PPV)
        if "d_ppv" in metrics:
            fm_df["d_ppv"] = fm_df["ppv"] - infoRoot["ppv"].values[0]

        if "d_tpr" in metrics:
            fm_df["d_tpr"] = fm_df["tpr"] - infoRoot["tpr"].values[0]
        if "d_tnr" in metrics:
            fm_df["d_tnr"] = fm_df["tnr"] - infoRoot["tnr"].values[0]
        if "d_npv" in metrics:
            fm_df["d_npv"] = fm_df["npv"] - infoRoot["npv"].values[0]
        if "d_fdr" in metrics:
            fm_df["d_fdr"] = fm_df["fdr"] - infoRoot["fdr"].values[0]
        if "d_for" in metrics:
            fm_df["d_for"] = fm_df["for"] - infoRoot["for"].values[0]

        ####### TO BE REMOVED IF NOT NECESSARY #########
        if "d_fnr_abs" in metrics:
            fm_df["d_fnr_abs"] = abs(fm_df["fnr"] - infoRoot["fnr"].values[0])
        if "d_fpr_abs" in metrics:
            fm_df["d_fpr_abs"] = abs(fm_df["fpr"] - infoRoot["fpr"].values[0])
        if "d_accuracy_abs" in metrics:
            fm_df["d_accuracy_abs"] = abs(
                fm_df["accuracy"] - infoRoot["accuracy"].values[0]
            )

        if "ACsf" in metrics:
            from .utils_metrics_FPx import AccuracySubgroupFairness

            fm_df = AccuracySubgroupFairness(fm_df)
        if "SPsf" in metrics:
            from .utils_metrics_FPx import statParitySubgroupFairness

            fm_df = statParitySubgroupFairness(fm_df)
        if "FPsf" in metrics:
            from .utils_metrics_FPx import FPSubgroupFairness

            fm_df = FPSubgroupFairness(fm_df)
        if "FNsf" in metrics:
            from .utils_metrics_FPx import FNSubgroupFairness

            fm_df = FNSubgroupFairness(fm_df)

        if "d_fnr_w" in metrics:
            alfaFN = (fm_df["tp"] + fm_df["fn"]) / infoRoot["support_count"].values[0]
            fm_df["d_fnr_w"] = alfaFN * fm_df["d_fnr"]
        if "d_fpr_w" in metrics:
            alfaFP = (fm_df["tn"] + fm_df["fp"]) / infoRoot["support_count"].values[0]
            fm_df["d_fpr_w"] = alfaFP * fm_df["d_fpr"]
        if "d_accuracy_w" in metrics:
            fm_df["d_accuracy_w"] = fm_df["support"] * fm_df["d_accuracy"]
        return fm_df

    def fpgrowth_divergence_metrics(
        self,
        df,
        df_confusion_matrix,
        min_support=0.5,
        use_colnames=False,
        verbose=0,
        cols_orderTP=["tn", "fp", "fn", "tp"],
        sortedV="support",
        attribute_id_mapping_for_compatibility=None,  ### Handling generalization/taxonomy
        save_in_progress=False,
        take_top_k=None,
        metric_top_k=None,
    ):
        from .utils_FPgrowth import fpgrowth_cm

        fp = fpgrowth_cm(
            df,
            df_confusion_matrix,
            min_support=min_support,
            use_colnames=use_colnames,
            cols_orderTP=cols_orderTP,
            attribute_id_mapping_for_compatibility=attribute_id_mapping_for_compatibility,
            save_in_progress=save_in_progress,
            take_top_k=take_top_k,
            metric_top_k=metric_top_k,
        )
        row_root = dict(df_confusion_matrix.sum())
        row_root.update({"support": 1, "itemsets": frozenset()})
        row_root = pd.DataFrame([row_root])
        fp = pd.concat([fp, row_root])
        fp["length"] = fp["itemsets"].str.len()

        fp["support_count"] = (fp["support"] * len(df)).round()

        # fp["fpr"]=fpr_df(fp[cols_orderTP])
        # fp["fnr"]=fnr_df(fp[cols_orderTP])
        # fp["accuracy"]=accuracy_df(fp[cols_orderTP])
        fp.sort_values(sortedV, ascending=False, inplace=True)
        fp = fp.reset_index(drop=True)
        return fp

    def getFrequentPatternDivergence(
        self,
        min_support,
        sortedV="support",
        metrics=["d_fpr", "d_fnr", "d_accuracy"],
        FPM_type="fpgrowth",
        viz_col=False,
        save_in_progress=False,
        take_top_k=None,
        metric_top_k=None,
    ):

        if take_top_k is not None:
            if metric_top_k is None:
                metric_top_k = metrics[0]
            if type(metric_top_k) != str:
                raise ValueError(
                    f"metric_top_k is the metric we optimize. {metric_top_k} was given"
                )
            if type(take_top_k) != int:
                raise ValueError(
                    f"take_top_k defines the top k to consider in the extraction process. {take_top_k} was given."
                )
        if (
            min_support in self.FP_metric_support
            and "FM" in self.FP_metric_support[min_support]
        ):
            return self.FP_metric_support[min_support]["FM"]

        y_conf_matrix = self.instanceConfusionMatrix(self.y_true_pred)

        if FPM_type not in ["fpgrowth", "apriori"]:
            raise ValueError(
                f'{FPM_type} algorithm is not handled. For now, we integrate the DivExplorer computation in "fpgrowth" and "apriori" algorithms.'
            )

        if FPM_type == "fpgrowth":
            conf_matrix_cols = ["tn", "fp", "fn", "tp"]
            ### Handling generalization/taxonomy
            # generalization = (
            #     self.generalizations_obj if self.hasGeneralization else None
            # )
            """
            incompatible_items = (
                list(self.incompatible_items.values())
                if self.hasGeneralization
                else None
            )
            """

            if self.hasGeneralization:
                attribute_id_mapping_for_compatibility = {}
                for attribute, incompatibilities in self.incompatible_items.items():
                    for item_id in incompatibilities:
                        attribute_id_mapping_for_compatibility[item_id] = attribute
            else:
                attribute_id_mapping_for_compatibility = None

            # TODO
            df_FP_metrics = self.fpgrowth_divergence_metrics(
                self.X.copy(),
                y_conf_matrix[conf_matrix_cols],
                min_support=min_support,
                use_colnames=True,
                sortedV=sortedV,
                attribute_id_mapping_for_compatibility=attribute_id_mapping_for_compatibility,
                save_in_progress=save_in_progress,
                take_top_k=take_top_k,
                metric_top_k=metric_top_k,
            )

        else:
            conf_matrix_cols = ["tp", "fp", "fn", "tn"]
            attributes_one_hot = self.X.columns
            df_with_conf_matrix = pd.concat(
                [self.X, y_conf_matrix[conf_matrix_cols]], axis=1
            )
            if self.hasGeneralization:
                from .utils_apriori import apriori_divergence_generalization_hierarchy

                df_FP_metrics = apriori_divergence_generalization_hierarchy(
                    df_with_conf_matrix[attributes_one_hot],
                    df_with_conf_matrix,
                    min_support=min_support,
                    use_colnames=True,
                    # generalization=self.generalizations_obj,
                    incompatible_items=self.incompatible_items,
                    save_in_progress=save_in_progress,
                    take_top_k=take_top_k,
                    metric_top_k=metric_top_k,
                )
            else:
                df_FP_metrics = self.apriori_divergence(
                    df_with_conf_matrix[attributes_one_hot],
                    df_with_conf_matrix,
                    min_support=min_support,
                    use_colnames=True,
                    sortedV=sortedV,
                )

        # TODO. In the case of take_top_k we are redoing it..
        df_FP_divergence = self.computeDivergenceItemsets(
            df_FP_metrics, metrics=metrics
        )

        if min_support not in self.FP_metric_support:
            self.FP_metric_support[min_support] = {}

        if viz_col:
            df_FP_divergence[VIZ_COL_NAME] = True
        self.FP_metric_support[min_support]["FM"] = df_FP_divergence

        # T_test values
        self.t_test_FP(min_support, metrics=metrics)

        return df_FP_divergence

    def mean_var_beta_distribution(self, metric, min_support):
        FPb = self.FP_metric_support[min_support]["FM"]
        cl_metric = map_beta_distribution[metric]
        FPb["a"] = 1 + FPb[cl_metric["T"]].sum(axis=1)
        FPb["b"] = 1 + FPb[cl_metric["F"]].sum(axis=1)
        cl_metric = "_".join(cl_metric["T"])
        FPb[f"mean_beta_{cl_metric}"] = _compute_mean_beta_distribution(FPb[["a", "b"]])
        FPb[f"var_beta_{cl_metric}"] = _compute_variance_beta_distribution(
            FPb[["a", "b"]]
        )
        FPb.drop(columns=["a", "b"], inplace=True)
        return FPb

    def t_test_FP(self, min_support, metrics=["d_fpr", "d_fnr", "d_accuracy"]):
        for metric in metrics:
            if metric not in map_beta_distribution:
                raise ValueError(f"{metric} not in {map_beta_distribution.keys()}")

            c_metric = "_".join(map_beta_distribution[metric]["T"])
            FPb = self.mean_var_beta_distribution(metric, min_support)

            mean_col, var_col = f"mean_beta_{c_metric}", f"var_beta_{c_metric}"
            mean_d, var_d = FPb.loc[FPb.itemsets == frozenset()][
                [mean_col, var_col]
            ].values[0]
            FPb[f"t_value_{c_metric}"] = _compute_t_test(
                FPb[[mean_col, var_col]], mean_col, var_col, mean_d, var_d
            )
            FPb.drop(
                columns=[f"mean_beta_{c_metric}", f"var_beta_{c_metric}"], inplace=True
            )
            self.FP_metric_support[min_support]["FM"] = FPb
        return FPb


def compute_divergence_itemsets(
    fm_df,
    metrics=["d_fpr", "d_fnr", "d_accuracy"],
    cols_orderTP=["tn", "fp", "fn", "tp"],
):

    # TODO - REFACTOR CODE

    if "d_fpr" in metrics:
        from .utils_metrics_FPx import fpr_df

        fm_df["fpr"] = fpr_df(fm_df[cols_orderTP])

    if "d_fnr" in metrics:
        from .utils_metrics_FPx import fnr_df

        fm_df["fnr"] = fnr_df(fm_df[cols_orderTP])

    if "d_accuracy" in metrics:
        from .utils_metrics_FPx import accuracy_df

        fm_df["accuracy"] = accuracy_df(fm_df[cols_orderTP])

    if "d_error" in metrics:
        from .utils_metrics_FPx import classification_error_df

        fm_df["error"] = classification_error_df(fm_df[cols_orderTP])

    if "d_ppv" in metrics:
        from .utils_metrics_FPx import positive_predicted_value_df

        fm_df["ppv"] = positive_predicted_value_df(fm_df[cols_orderTP])

    if "d_tpr" in metrics:
        from .utils_metrics_FPx import true_positive_rate_df

        fm_df["tpr"] = true_positive_rate_df(fm_df[cols_orderTP])

    if "d_tnr" in metrics:
        from .utils_metrics_FPx import true_negative_rate_df

        fm_df["tnr"] = true_negative_rate_df(fm_df[cols_orderTP])

    if "d_npv" in metrics:
        from .utils_metrics_FPx import negative_predicted_value_df

        fm_df["npv"] = negative_predicted_value_df(fm_df[cols_orderTP])

    if "d_fdr" in metrics:
        from .utils_metrics_FPx import false_discovery_rate_df

        fm_df["fdr"] = false_discovery_rate_df(fm_df[cols_orderTP])

    if "d_for" in metrics:
        from .utils_metrics_FPx import false_omission_rate_df

        fm_df["for"] = false_omission_rate_df(fm_df[cols_orderTP])

    if "d_posr" in metrics:
        # TODO
        from .utils_metrics_FPx import getInfoRoot

        rootIndex = getInfoRoot(fm_df).index
        from .utils_metrics_FPx import get_pos, posr_df

        fm_df["P"] = get_pos(fm_df[cols_orderTP])
        fm_df["posr"] = posr_df(fm_df[cols_orderTP])
        fm_df["d_posr"] = fm_df["posr"] - fm_df.loc[rootIndex]["posr"].values[0]
    if "d_negr" in metrics:
        # TODO
        from .utils_metrics_FPx import getInfoRoot

        rootIndex = getInfoRoot(fm_df).index
        from .utils_metrics_FPx import get_neg, negr_df

        fm_df["N"] = get_neg(fm_df[cols_orderTP])
        fm_df["negr"] = negr_df(fm_df[cols_orderTP])
        fm_df["d_negr"] = fm_df["negr"] - fm_df.loc[rootIndex]["negr"].values[0]

    from .utils_metrics_FPx import getInfoRoot

    infoRoot = getInfoRoot(fm_df)

    if "d_fnr" in metrics:
        fm_df["d_fnr"] = fm_df["fnr"] - infoRoot["fnr"].values[0]
    if "d_fpr" in metrics:
        fm_df["d_fpr"] = fm_df["fpr"] - infoRoot["fpr"].values[0]
    if "d_accuracy" in metrics:
        fm_df["d_accuracy"] = fm_df["accuracy"] - infoRoot["accuracy"].values[0]

    # Classification error
    if "d_error" in metrics:
        fm_df["d_error"] = fm_df["error"] - infoRoot["error"].values[0]

    # Precision or positive predictive value (PPV)
    if "d_ppv" in metrics:
        fm_df["d_ppv"] = fm_df["ppv"] - infoRoot["ppv"].values[0]

    if "d_tpr" in metrics:
        fm_df["d_tpr"] = fm_df["tpr"] - infoRoot["tpr"].values[0]
    if "d_tnr" in metrics:
        fm_df["d_tnr"] = fm_df["tnr"] - infoRoot["tnr"].values[0]
    if "d_npv" in metrics:
        fm_df["d_npv"] = fm_df["npv"] - infoRoot["npv"].values[0]
    if "d_fdr" in metrics:
        fm_df["d_fdr"] = fm_df["fdr"] - infoRoot["fdr"].values[0]
    if "d_for" in metrics:
        fm_df["d_for"] = fm_df["for"] - infoRoot["for"].values[0]

    ####### TO BE REMOVED IF NOT NECESSARY #########
    if "d_fnr_abs" in metrics:
        fm_df["d_fnr_abs"] = abs(fm_df["fnr"] - infoRoot["fnr"].values[0])
    if "d_fpr_abs" in metrics:
        fm_df["d_fpr_abs"] = abs(fm_df["fpr"] - infoRoot["fpr"].values[0])
    if "d_accuracy_abs" in metrics:
        fm_df["d_accuracy_abs"] = abs(
            fm_df["accuracy"] - infoRoot["accuracy"].values[0]
        )

    if "ACsf" in metrics:
        from .utils_metrics_FPx import AccuracySubgroupFairness

        fm_df = AccuracySubgroupFairness(fm_df)
    if "SPsf" in metrics:
        from .utils_metrics_FPx import statParitySubgroupFairness

        fm_df = statParitySubgroupFairness(fm_df)
    if "FPsf" in metrics:
        from .utils_metrics_FPx import FPSubgroupFairness

        fm_df = FPSubgroupFairness(fm_df)
    if "FNsf" in metrics:
        from .utils_metrics_FPx import FNSubgroupFairness

        fm_df = FNSubgroupFairness(fm_df)

    if "d_fnr_w" in metrics:
        alfaFN = (fm_df["tp"] + fm_df["fn"]) / infoRoot["support_count"].values[0]
        fm_df["d_fnr_w"] = alfaFN * fm_df["d_fnr"]
    if "d_fpr_w" in metrics:
        alfaFP = (fm_df["tn"] + fm_df["fp"]) / infoRoot["support_count"].values[0]
        fm_df["d_fpr_w"] = alfaFP * fm_df["d_fpr"]
    if "d_accuracy_w" in metrics:
        fm_df["d_accuracy_w"] = fm_df["support"] * fm_df["d_accuracy"]
    return fm_df


def instanceConfusionMatrix(df, class_name='class', predicted_name = 'predicted', class_map = {0: 'N', 1: 'P' }):
    # TODO
    df["tn"] = (
        (df[class_name] == df[predicted_name]) & (df[class_name] == class_map["N"])
    ).astype(int)
    df["fp"] = (
        (df[class_name] != df[predicted_name]) & (df[class_name] == class_map["N"])
    ).astype(int)
    df["tp"] = (
        (df[class_name] == df[predicted_name]) & (df[class_name] == class_map["P"])
    ).astype(int)
    df["fn"] = (
        (df[class_name] != df[predicted_name]) & (df[class_name] == class_map["P"])
    ).astype(int)
    return df



def prune_categorical(df_input, categorical_attributes, metric, true_class_name = 'class', pred_class_name = 'predicted', cols_orderTP = ["tn", "fp", "fn", "tp"], class_map = {'N': 0, 'P': 1}):
    import pandas as pd
    
    import numpy as np
    
    from copy import deepcopy
    if metric == 'd_outcome':
        raise ValueError()
        
    y_true_pred = deepcopy(df_input[[true_class_name, pred_class_name]])

    #from divexplorer_generalized.FP_DivergenceExplorer import instanceConfusionMatrix

    targets = instanceConfusionMatrix(y_true_pred, true_class_name, pred_class_name, class_map)

    #from divexplorer_generalized.FP_DivergenceExplorer import oneHotEncoding

    X_cat_onh = oneHotEncoding(df_input[categorical_attributes])
    
    
    df_with_conf_matrix = pd.concat(
                [X_cat_onh, targets[cols_orderTP]], axis=1
            )
    
    ary_col_idx = np.arange(X_cat_onh.values.shape[1])
    itemset_dict = ary_col_idx.reshape(-1, 1)
    
    
    def filterColumns(df_filter, cols):
        return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]

    def sum_values(_x):
        out = np.sum(_x, axis=0)
        return np.array(out).reshape(-1)

    targets_sum = [sum_values(filterColumns(df_with_conf_matrix, item)[cols_orderTP]) for item in itemset_dict]
    targets_sum.append(np.asarray([sum_values(df_with_conf_matrix[cols_orderTP])][0]))

    df1 = pd.DataFrame(targets_sum, columns=cols_orderTP)
    df1['itemsets'] = list(X_cat_onh.columns) + [frozenset([])]
    
    from divexplorer_generalized.FP_DivergenceExplorer import compute_divergence_itemsets

    pattern_divergence = compute_divergence_itemsets(df1, metrics = [metric])
    items = list(pattern_divergence.loc[pattern_divergence[metric]>0]['itemsets'].values)

    keep_items = {}
    for item in items:
        s = item.split('=')
        attribute, value = s[0], "=".join(s[1:])
        if attribute not in keep_items:
            keep_items[attribute] = []
        keep_items[attribute].append(value)

    return keep_items




def prune_categorical2(df_input, categorical_attributes, metric, true_class_name = 'class', pred_class_name = 'predicted', cols_orderTP = ["tn", "fp", "fn", "tp"], class_map = {'N': 0, 'P': 1}):
    import pandas as pd
    
    import numpy as np
    
    from copy import deepcopy
    if metric == 'd_outcome':
        raise ValueError()
        
    y_true_pred = deepcopy(df_input[[true_class_name, pred_class_name]])

    #from divexplorer_generalized.FP_DivergenceExplorer import instanceConfusionMatrix

    targets = instanceConfusionMatrix(y_true_pred, true_class_name, pred_class_name, class_map)

    #from divexplorer_generalized.FP_DivergenceExplorer import oneHotEncoding

    X_cat_onh = oneHotEncoding(df_input[categorical_attributes])
    
    
    df_with_conf_matrix = pd.concat(
                [X_cat_onh, targets[cols_orderTP]], axis=1
            )
    
    ary_col_idx = np.arange(X_cat_onh.values.shape[1])
    itemset_dict = ary_col_idx.reshape(-1, 1)
    
    
    def filterColumns(df_filter, cols):
        return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]

    def sum_values(_x):
        out = np.sum(_x, axis=0)
        return np.array(out).reshape(-1)

    targets_sum = [sum_values(filterColumns(df_with_conf_matrix, item)[cols_orderTP]) for item in itemset_dict]
    targets_sum.append(np.asarray([sum_values(df_with_conf_matrix[cols_orderTP])][0]))

    df1 = pd.DataFrame(targets_sum, columns=cols_orderTP)
    df1['itemsets'] = list(X_cat_onh.columns) + [frozenset([])]

    from divexplorer_generalized.FP_DivergenceExplorer import compute_divergence_itemsets

    pattern_divergence = compute_divergence_itemsets(df1, metrics = [metric])
    items = list(pattern_divergence.loc[pattern_divergence[metric]>0]['itemsets'].values)

    keep_items = {}
    for item in items:
        s = item.split('=')
        attribute, value = s[0], "=".join(s[1:])
        if attribute not in keep_items:
            keep_items[attribute] = []
        keep_items[attribute].append(value)

    return keep_items
