import numpy as np

"""

def tpr_df(df_cm):
    return df_cm["tp"]/(df_cm["tp"]+df_cm["fn"])

def fpr_df(df_cm):
    return df_cm["fp"]/(df_cm["fp"]+df_cm["tn"])

def fnr_df(df_cm):
    return df_cm["fn"]/(df_cm["tp"]+df_cm["fn"])

def tnr_df(df_cm):
    return df_cm["tn"]/(df_cm["fp"]+df_cm["tn"])

"""


def tpr_df(df_cm):
    return (df_cm["tp"] / (df_cm["tp"] + df_cm["fn"])).fillna(0)


def fpr_df(df_cm):
    return (df_cm["fp"] / (df_cm["fp"] + df_cm["tn"])).fillna(0)


def fnr_df(df_cm):
    return (df_cm["fn"] / (df_cm["tp"] + df_cm["fn"])).fillna(0)


def tnr_df(df_cm):
    return (df_cm["tn"] / (df_cm["fp"] + df_cm["tn"])).fillna(0)


def accuracy_df(df_cm):
    return (df_cm["tn"] + df_cm["tp"]) / (
        df_cm["tp"] + df_cm["fp"] + df_cm["tn"] + df_cm["fn"]
    )


def getInfoRoot(df, orient=None):
    return df.loc[df["itemsets"] == frozenset()]


def statParitySubgroupFairness(df):
    root_info = getInfoRoot(df)
    alfaSP = df["support"]
    SP_D = (root_info["tp"].values[0] + root_info["fp"].values[0]) / root_info[
        "support_count"
    ].values[0]
    SP_DG = (df["tp"] + df["fp"]) / df["support_count"]
    betaSP = abs(SP_D - SP_DG)
    df["SPsf"] = alfaSP * betaSP
    return df


def FPSubgroupFairness(df):
    root_info = getInfoRoot(df)
    alfaFP = (df["tn"] + df["fp"]) / root_info["support_count"].values[0]
    FP_D = root_info["fp"].values[0] / (
        root_info["fp"].values[0] + root_info["tn"].values[0]
    )
    # Redundant, equalt to fpr_rate
    FP_DG = (df["fp"]) / (df["fp"] + df["tn"])
    betaFP = abs(FP_D - FP_DG)
    df["FPsf"] = alfaFP * betaFP
    return df


def FNSubgroupFairness(df):
    root_info = getInfoRoot(df)
    alfaFN = (df["tp"] + df["fn"]) / root_info["support_count"].values[0]
    FN_D = root_info["fn"].values[0] / (
        root_info["fn"].values[0] + root_info["tp"].values[0]
    )
    # Redundant, equalt to fnr_rate
    FN_DG = (df["fn"]) / (df["fn"] + df["tp"])
    betaFN = abs(FN_D - FN_DG)
    df["FNsf"] = alfaFN * betaFN
    return df


def getAccuracyDF(df):
    df["accuracy"] = (df["tp"] + df["tn"]) / (df["tp"] + df["tn"] + df["fn"] + df["fp"])
    return df


def AccuracySubgroupFairness(df):
    if "accuracy" not in df.columns:
        df = getAccuracyDF(df)
    root_info = getInfoRoot(df)
    alfaAC = df["support"]
    AC_D = (root_info["tp"].values[0] + root_info["tn"].values[0]) / (
        root_info["support_count"].values[0]
    )
    AC_DG = df["accuracy"]
    if "d_accuracy" not in df:
        df["d_accuracy"] = df["accuracy"] - AC_D
        # df["d_accuracy_abs"]=abs(df["accuracy"]-AC_D)

    betaAC = abs(AC_D - AC_DG)
    df["ACsf"] = alfaAC * betaAC
    return df


def computeInstancesLogLoss(y, y_predict_prob):
    from sklearn.metrics import log_loss

    labels = np.unique(y).copy()
    return [
        log_loss([y.values[i]], [y_predict_prob[i]], labels=labels)
        for i in range(0, len(y))
    ]


def computeDfInstancesLogLoss(X, y, clf):
    from sklearn.metrics import log_loss

    y_p = clf.predict_proba(X)
    return [
        log_loss([y.values[i]], [y_p[i]], labels=clf.classes_) for i in range(0, len(y))
    ]


def effect_size(sample_a, reference):
    import math

    mu, s, n = reference[0], reference[1], reference[2]
    if n - len(sample_a) == 0:
        return 0
    sample_b_mean = (mu * n - np.sum(sample_a)) / (n - len(sample_a))
    sample_b_var = (s ** 2 * (n - 1) - np.std(sample_a) ** 2 * (len(sample_a) - 1)) / (
        n - len(sample_a) - 1
    )
    if sample_b_var < 0:
        sample_b_var = 0.0

    diff = np.mean(sample_a) - sample_b_mean
    diff /= math.sqrt((np.std(sample_a) + math.sqrt(sample_b_var)) / 2.0)
    return diff
