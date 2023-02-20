# Name for diverge in the paper
div_name = "Δ"


def printable(
    df_print,
    cols=["itemsets"],
    abbreviations={},
    decimals=(2, 3),
    resort_cols=True,
):

    if type(decimals) is tuple:
        r1, r2 = decimals[0], decimals[1]
    else:
        r1, r2 = decimals, decimals
    df_print = df_print.copy()
    if "support" in df_print.columns:
        df_print["support"] = df_print["support"].round(r1)
    t_v = [c for c in df_print.columns if "t_value_" in c]
    if t_v:
        df_print[t_v] = df_print[t_v].round(1)
    df_print = df_print.round(r2)
    df_print.rename(columns={"support": "sup"}, inplace=True)
    df_print.columns = df_print.columns.str.replace("d_*", f"{div_name}_", regex=True)
    df_print.columns = df_print.columns.str.replace("t_value", "t", regex=True)
    for c in cols:
        df_print[c] = df_print[c].apply(lambda x: sortItemset(x, abbreviations))

    cols = list(df_print.columns)

    if resort_cols:
        itemset_col = "itemsets"
        if itemset_col in cols:
            cols.remove(itemset_col)
            cols = [itemset_col] + cols

    return df_print[cols]


def sortItemset(x, abbreviations={}):
    x = list(x)
    x.sort()
    x = ", ".join(x)
    for k, v in abbreviations.items():
        x = x.replace(k, v)
    return x


def highlight_max(data, use_bold=True, color="yellow"):

    # Highlight in bold or with a different columns the maximum in a Series (or DataFrame)

    if use_bold:
        attr = "font-weight: bold"
    else:
        attr = "background-color: {}".format(color)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else "" for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        import numpy as np
        import pandas as pd

        return pd.DataFrame(
            np.where(is_max, attr, ""), index=data.index, columns=data.columns
        )
