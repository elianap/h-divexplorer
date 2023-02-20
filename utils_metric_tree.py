import numpy as np


map_beta_distribution = {
    "d_fpr": {"T": ["fp"], "F": ["tn"]},
    "d_fnr": {"T": ["fn"], "F": ["tp"]},
    # "d_accuracy": {"T": ["tp", "tn"], "F": ["fp", "fn"]},
    # "d_fpr_abs": {"T": ["fp"], "F": ["tn"]},
    # "d_fnr_abs": {"T": ["fn"], "F": ["tp"]},
    # "d_accuracy_abs": {"T": ["tp", "tn"], "F": ["fp", "fn"]},
    # "d_posr": {"T": ["tp", "fn"], "F": ["tn", "fp"]},
    # "d_negr": {"T": ["tn", "fp"], "F": ["tp", "fn"]},
    # "d_classiferror": {"T": ["tp", "tn"], "F": ["fp", "fn"]},
    # "d_classiferror_abs": {"T": ["tp", "tn"], "F": ["fp", "fn"]},
}


def tpr_df(df_cm):
    # TODO
    if (df_cm["tp"] + df_cm["fn"]) == 0:
        return 0.0
    return df_cm["tp"] / (df_cm["tp"] + df_cm["fn"])


def fpr_df(df_cm):
    # TODO
    if (df_cm["fp"] + df_cm["tn"]) == 0:
        return 0.0
    return df_cm["fp"] / (df_cm["fp"] + df_cm["tn"])


def fnr_df(df_cm):
    if (df_cm["tp"] + df_cm["fn"]) == 0:
        return 0.0
    return df_cm["fn"] / (df_cm["tp"] + df_cm["fn"])


def tnr_df(df_cm):
    if (df_cm["fp"] + df_cm["tn"]) == 0:
        return 0.0
    return df_cm["tn"] / (df_cm["fp"] + df_cm["tn"])


def accuracy_df(df_cm):
    return (df_cm["tn"] + df_cm["tp"]) / (
        df_cm["tp"] + df_cm["fp"] + df_cm["tn"] + df_cm["fn"]
    )


def error_df(df_cm):
    return (df_cm["fn"] + df_cm["fp"]) / (
        df_cm["tp"] + df_cm["fp"] + df_cm["tn"] + df_cm["fn"]
    )


def sum_values(_x):
    out = np.sum(_x, axis=0)
    return np.array(out).reshape(-1)


def getConfusionMatrix(data_i, cols_orderTP=["tn", "fp", "fn", "tp"]):
    conf_matrix_i = sum_values(data_i[cols_orderTP])
    conf_matrix_i = {
        cols_orderTP[i]: conf_matrix_i[i] for i in range(0, len(cols_orderTP))
    }
    return conf_matrix_i


def getDivergenceMetric(
    data_sel, metric="d_fpr", cols_orderTP=["tn", "fp", "fn", "tp"], metric_baseline=0
):
    performanceMetricSlice = getPerformanceMetric(
        data_sel, metric=metric, cols_orderTP=cols_orderTP
    )
    if performanceMetricSlice is not None:
        return performanceMetricSlice - metric_baseline
    else:
        raise ValueError("-------------------")


def getPerformanceMetric(
    data_sel, metric="d_fpr", cols_orderTP=["tn", "fp", "fn", "tp"]
):
    sums = getConfusionMatrix(data_sel)
    if metric == "d_fpr":
        return fpr_df(sums)
    elif metric == "d_fnr":
        return fnr_df(sums)
    elif metric == "d_accuracy":
        import warnings

        # TODO
        # warnings.warn("Check accuracy - TODO")
        return accuracy_df(sums)
    elif metric == "d_error":
        import warnings

        # TODO
        # warnings.warn("Check error - TODO")
        return error_df(sums)
    else:
        # TODO
        raise ValueError("TODO - add other metrics")
        # return None


def getSupport(data_sel, dataset_len):
    return len(data_sel) / dataset_len


def check_valid_in_list(par, list_par):
    if par not in list_par:
        raise ValueError(f"{par} not in {list_par}")


# TODO
# Other criteria, e.g.:
# weighted absolute sum --> to also consider the size/support of the split


def evaluateCriterion(
    val_n1,
    val_n2,
    type_criterion="sum_abs",
    sup_n1=None,
    sup_n2=None,
    baseline=0,
    verbose=False,
):

    check_valid_in_list(
        type_criterion,
        [
            "sum_abs",
            "sum_pow",
            "pow_sum",
            "max_abs",
            "weighted_sum_abs",
            "weighted_sum_pow",
            "weighted_max_abs",
            "weighted_KL",
        ],
    )
    if type_criterion == "sum_abs":
        return abs(val_n1) + abs(val_n2)
    elif type_criterion == "pow_sum":
        return (val_n1 + val_n2) ** 2
    elif type_criterion == "sum_pow":
        return (val_n1 ** 2) + (val_n2 ** 2)
    elif type_criterion == "max_abs":
        return max(abs(val_n1), abs(val_n2))
    elif type_criterion == "weighted_sum_abs":
        if sup_n1 is None or sup_n2 is None:
            raise ValueError("Support values are not provided")
        return sup_n1 * abs(val_n1) + sup_n2 * abs(val_n2)
    elif type_criterion == "weighted_sum_pow":
        if sup_n1 is None or sup_n2 is None:
            raise ValueError("Support values are not provided")
        return sup_n1 * ((val_n1) ** 2) + sup_n2 * ((val_n2) ** 2)
    elif type_criterion == "weighted_max_abs":
        if sup_n1 is None or sup_n2 is None:
            raise ValueError("Support values are not provided")
        return max(sup_n1 * abs(val_n1), sup_n2 * abs(val_n2))
    elif type_criterion == "weighted_max_pow":
        if sup_n1 is None or sup_n2 is None:
            raise ValueError("Support values are not provided")
        return max(sup_n1 * (val_n1) ** 2, sup_n2 * (val_n2) ** 2)
    elif type_criterion == "weighted_KL":
        if sup_n1 is None or sup_n2 is None:
            raise ValueError("Support values are not provided")
        import math

        def val_i_KL(val, baseline):
            if val == 0:
                return 0
            else:
                return abs(math.log(val / baseline, 2))

        if verbose:
            print(sup_n1, sup_n2, val_n1, val_n2, baseline)
            print(
                sup_n1 * val_i_KL(val_n1, baseline)
                + sup_n2 * val_i_KL(val_n2, baseline)
            )

        return sup_n1 * val_i_KL(val_n1, baseline) + sup_n2 * val_i_KL(val_n2, baseline)


def isAttributeBinary(attr_vals):
    return len(attr_vals) == 2


def getReciprocal(attr_vals, val):
    return [v for v in attr_vals.tolist() if v != val][0]


def instanceConfusionMatrix(df, class_map, class_name="class", pred_name="predicted"):
    # TODO
    df_cm = df.copy()
    df_cm["tn"] = (
        (df_cm[class_name] == df_cm[pred_name]) & (df_cm[class_name] == class_map["N"])
    ).astype(int)
    df_cm["fp"] = (
        (df_cm[class_name] != df_cm[pred_name]) & (df_cm[class_name] == class_map["N"])
    ).astype(int)
    df_cm["tp"] = (
        (df_cm[class_name] == df_cm[pred_name]) & (df_cm[class_name] == class_map["P"])
    ).astype(int)
    df_cm["fn"] = (
        (df_cm[class_name] != df_cm[pred_name]) & (df_cm[class_name] == class_map["P"])
    ).astype(int)
    return df_cm


def _compute_t_test(df, col_mean, col_var, mean_d, var_d):
    return (abs(df[col_mean] - mean_d)) / ((df[col_var] + var_d) ** 0.5)


# def _compute_std_beta_distribution(FPb):
#    return ((FPb.a*FPb.b)/((FPb.a+FPb.b)**2*(FPb.a+FPb.b+1)))**(1/2)


def _compute_variance_beta_distribution(FPb):
    return (FPb.a * FPb.b) / ((FPb.a + FPb.b) ** 2 * (FPb.a + FPb.b + 1))


def _compute_mean_beta_distribution(FPb):
    return FPb.a / (FPb.a + FPb.b)


def evaluate_KL(p, q):
    def p_log_q_over_p(p, q):
        if p == 0:
            return 0
        if q == 0:
            # TODO
            return 0
        from math import log

        return -p * log(q / p)

    return p_log_q_over_p(p, q) + p_log_q_over_p(1 - p, 1 - q)


def evaluate_KL_derived(w, p, q):
    def log_p_over_q(p, q):
        if p == 0:
            return 0
        if q == 0:
            # TODO
            return 0
        from math import log

        return -log(p / q)

    from TreeDivergence import PRINT_VERBOSE

    # if PRINT_VERBOSE:
    #    print("---p", p, "q", q, w)
    #    print(log_p_over_q(p, q), "w", w, 'res:', w * (log_p_over_q(p, q)) )
    return w * (log_p_over_q(p, q) + log_p_over_q(1 - p, 1 - q))


def evaluate_log_ratio(p, q):
    if p == 0:
        raise ValueError()
        # return 0
    if q == 0:
        # TODO
        # return 0
        raise ValueError()
    from math import log

    return log(p / q, 2)


def evaluate_log_ratio_cap(p, q):
    from math import log

    if p == 0:
        # raise ValueError()
        return 0
    if q == 0:
        # TODO
        # raise ValueError()

        return 0
    ratio = (p) / q

    if ratio < 1:
        return log(2 - (ratio), 2)
    else:
        return log(ratio, 2)


def evaluate_KL_derived_split(w1, p1, w2, p2, q):
    return evaluate_KL_derived(w1, p1, q) + evaluate_KL_derived(w2, p2, q)


def evaluate_KL_split(p1_i, p1_tot, p2_i, p2_tot, baseline):

    p1 = p1_i / p1_tot
    p2 = p2_i / p2_tot
    tot = p2_tot + p2_tot
    kl_node_1 = evaluate_KL(p1, baseline)
    kl_node_2 = evaluate_KL(p2, baseline)
    return (p1_tot / tot) * kl_node_1 + (p2_tot / tot) * kl_node_2


def entropy(p):
    if p == 0:
        return 0
    from math import log

    return -p * log(p, 2)


def evaluate_entropy_split(p1_i, p1_tot, p2_i, p2_tot, baseline, verbose=False):

    p1 = p1_i / p1_tot
    p2 = p2_i / p2_tot

    tot = p1_tot + p2_tot
    entropy_node_1 = entropy(p1) + entropy(1 - p1)
    entropy_node_2 = entropy(p2) + entropy(1 - p2)

    return (p1_tot / tot) * entropy_node_1 + (p2_tot / tot) * entropy_node_2


def entropy_node(data_node, metric="d_fpr"):

    p_i, p_tot = get_n_i_over_n(data_node, metric=metric)
    return entropy(p_i / p_tot) + entropy(1 - (p_i / p_tot))


def get_measure_node(data_node, metric="d_fpr"):
    p_i, p_tot = get_n_i_over_n(data_node, metric=metric)
    return p_i / p_tot


def evaluate_split_divergence(
    data_val_1,
    data_val_2,
    baseline,
    metric="d_fpr",
    type_criterion="sum_abs",
    sup_n1=1,
    sup_n2=1,
    p_k_root=None,
    p_root=None,
    verbose=False,
):
    if type_criterion == "KL":
        from utils_metric_tree import get_n_i_over_n

        p1_i, p1_tot = get_n_i_over_n(data_val_1, metric=metric)
        p2_i, p2_tot = get_n_i_over_n(data_val_2, metric=metric)
        criteria = evaluate_KL_split(p1_i, p1_tot, p2_i, p2_tot, baseline)
        divergence_1 = p1_i / p1_tot - baseline
        divergence_2 = p2_i / p2_tot - baseline

    elif type_criterion == "entropy":
        from utils_metric_tree import get_n_i_over_n

        p1_i, p1_tot = get_n_i_over_n(data_val_1, metric=metric)
        p2_i, p2_tot = get_n_i_over_n(data_val_2, metric=metric)

        criteria = evaluate_entropy_split(
            p1_i, p1_tot, p2_i, p2_tot, baseline, verbose=verbose
        )
        divergence_1 = p1_i / p1_tot - baseline
        divergence_2 = p2_i / p2_tot - baseline
    elif type_criterion == "KL_derived":
        from utils_metric_tree import get_n_i_over_n

        p1_i, p1_tot = get_n_i_over_n(data_val_1, metric=metric)
        p2_i, p2_tot = get_n_i_over_n(data_val_2, metric=metric)

        criteria = evaluate_KL_derived_split(
            (p1_tot + 1) / (p_root + 1),
            (p1_i + 1) / (p1_tot + 1),
            (p2_tot + 1) / (p_root + 1),
            (p2_i + 1) / (p2_tot + 1),
            (p_k_root + 1) / (p_root + 1),
        )
        divergence_1 = p1_i / p1_tot - baseline
        divergence_2 = p2_i / p2_tot - baseline
    else:
        from utils_metric_tree import getDivergenceMetric_np

        divergence_1 = getDivergenceMetric_np(
            data_val_1,
            metric=metric,
            metric_baseline=baseline,
        )
        divergence_2 = getDivergenceMetric_np(
            data_val_2,
            metric=metric,
            metric_baseline=baseline,
        )
        metric_1, metric_2 = divergence_1, divergence_2

        if type_criterion == "weighted_KL":
            # TO DO
            metric_1 += baseline
            metric_2 += baseline

        criteria = evaluateCriterion(
            metric_1,
            metric_2,
            type_criterion=type_criterion,
            sup_n1=sup_n1,
            sup_n2=sup_n2,
            baseline=baseline,
            verbose=verbose,
        )

    return criteria, divergence_1, divergence_2


# TODO - DIFFERENT from DivExplorer
def mean_var_beta_distribution(FPb, metric):
    cl_metric = map_beta_distribution[metric]
    FPb["a"] = 1 + FPb[cl_metric["T"]].sum(axis=1)
    FPb["b"] = 1 + FPb[cl_metric["F"]].sum(axis=1)
    cl_metric = "_".join(cl_metric["T"])
    FPb[f"mean_beta_{cl_metric}"] = _compute_mean_beta_distribution(FPb[["a", "b"]])
    FPb[f"var_beta_{cl_metric}"] = _compute_variance_beta_distribution(FPb[["a", "b"]])
    FPb.drop(columns=["a", "b"], inplace=True)
    return FPb


# TODO - DIFFERENT from DivExplorer
def t_test_FP(FPb, metrics=["d_fpr", "d_fnr", "d_accuracy"]):
    for metric in metrics:
        if metric not in map_beta_distribution:
            raise ValueError(f"{metric} not in {map_beta_distribution.keys()}")

        c_metric = "_".join(map_beta_distribution[metric]["T"])
        FPb = mean_var_beta_distribution(FPb, metric)

        mean_col, var_col = f"mean_beta_{c_metric}", f"var_beta_{c_metric}"
        mean_d, var_d = FPb.loc[FPb.itemset == frozenset()][[mean_col, var_col]].values[
            0
        ]
        FPb[f"t_value_{c_metric}"] = _compute_t_test(
            FPb[[mean_col, var_col]], mean_col, var_col, mean_d, var_d
        )
        FPb.drop(
            columns=[f"mean_beta_{c_metric}", f"var_beta_{c_metric}"], inplace=True
        )
    return FPb


def getDivergenceMetric_np(
    conf_m,
    metric="d_fpr",
    cols_orderTP_dict={"tn": 0, "fp": 1, "fn": 2, "tp": 3},
    metric_baseline=0,
):
    performanceMetricSlice = getPerformanceMetric_np(
        conf_m, metric=metric, cols_orderTP_dict=cols_orderTP_dict
    )
    if performanceMetricSlice is not None:
        return performanceMetricSlice - metric_baseline
    else:
        raise ValueError("-------------------")


def getPerformanceMetric_np(
    conf_m, metric="d_fpr", cols_orderTP_dict={"tn": 0, "fp": 1, "fn": 2, "tp": 3}
):
    if metric == "d_fpr":
        num_v = ["fp"]
        den_v = ["fp", "tn"]
    elif metric == "d_fnr":
        num_v = ["fn"]
        den_v = ["fn", "tp"]
    elif metric == "d_accuracy":
        import warnings

        # TODO
        # warnings.warn("Check accuracy - TODO")
        num_v = ["tp", "tn"]
        den_v = ["tp", "tn", "fp", "fn"]
    elif metric == "d_error":
        import warnings

        # TODO
        # warnings.warn("Check error - TODO")
        num_v = ["fp", "fn"]
        den_v = ["tp", "tn", "fp", "fn"]
    else:
        # TODO
        raise ValueError("TODO - add other metrics")
    num_i = [cols_orderTP_dict[k] for k in num_v]
    den_i = [cols_orderTP_dict[k] for k in den_v]

    den_sum = np.sum(conf_m[:, den_i])

    if den_sum == 0:
        return 0.0
    num_sum = np.sum(conf_m[:, num_i])
    return num_sum / den_sum


def get_n_i_over_n(
    conf_m, metric="d_fpr", cols_orderTP_dict={"tn": 0, "fp": 1, "fn": 2, "tp": 3}
):
    if metric == "d_fpr":
        num_v = ["fp"]
        den_v = ["fp", "tn"]
    elif metric == "d_fnr":
        num_v = ["fn"]
        den_v = ["fn", "tp"]
    elif metric == "d_accuracy":
        import warnings

        # TODO
        # warnings.warn("Check accuracy - TODO")
        num_v = ["tp", "tn"]
        den_v = ["tp", "tn", "fp", "fn"]
    elif metric == "d_error":
        import warnings

        # TODO
        # warnings.warn("Check error - TODO")
        num_v = ["fp", "fn"]
        den_v = ["tp", "tn", "fp", "fn"]
    else:
        # TODO
        raise ValueError("TODO - add other metrics")

    num_i = [cols_orderTP_dict[k] for k in num_v]
    den_i = [cols_orderTP_dict[k] for k in den_v]

    num_sum = np.sum(conf_m[:, num_i])
    den_sum = np.sum(conf_m[:, den_i])

    return num_sum, den_sum


def getConfusionMatrix_np(
    conf_m, cols_orderTP_dict={"tn": 0, "fp": 1, "fn": 2, "tp": 3}
):
    conf_matrix_i = np.sum(conf_m, axis=0)

    conf_matrix_i = {
        name: conf_matrix_i[id_col] for name, id_col in cols_orderTP_dict.items()
    }
    return conf_matrix_i
