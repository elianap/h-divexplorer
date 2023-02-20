import numpy as np
import pandas as pd
import collections
from distutils.version import LooseVersion as Version
from pandas import __version__ as pandas_version


def filterColumns(df_filter, cols):
    return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]


def sum_values(_x):
    out = np.sum(_x, axis=0)
    return np.array(out).reshape(-1)


def setup_fptree(df, min_support, cm):
    num_itemsets = len(df.index)  # number of itemsets in the database

    is_sparse = False
    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            itemsets = df.values
        else:
            itemsets = df.sparse.to_coo().tocsr()
            is_sparse = True
    else:
        # dense DataFrame
        itemsets = df.values

    # support of each individual item
    # if itemsets is sparse, np.sum returns an np.matrix of shape (1, N)
    item_support = np.array(np.sum(itemsets, axis=0) / float(num_itemsets))
    item_support = item_support.reshape(-1)

    items = np.nonzero(item_support >= min_support)[0]

    # Define ordering on items for inserting into FPTree
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}

    if is_sparse:
        # Ensure that there are no zeros in sparse DataFrame
        itemsets.eliminate_zeros()

    # Building tree by inserting itemsets in sorted order
    # Heuristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = FPTree(rank)
    for i in range(num_itemsets):
        if is_sparse:
            # itemsets has been converted to CSR format to speed-up the line
            # below.  It has 3 attributes:
            #  - itemsets.data contains non null values, shape(#nnz,)
            #  - itemsets.indices contains the column number of non null
            #    elements, shape(#nnz,)
            #  - itemsets.indptr[i] contains the offset in itemset.indices of
            #    the first non null element in row i, shape(1+#nrows,)
            nonnull = itemsets.indices[itemsets.indptr[i] : itemsets.indptr[i + 1]]
        else:
            nonnull = np.where(itemsets[i, :])[0]
        # itemset = [item for item in nonnull if item in rank]
        itemset = list(set(nonnull).intersection(rank))
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset, cm_i=cm[i].copy())

    return tree, rank


# def write_csv_in_chunck(df_chunck, filename, i, n=1000):
#     if i == 1:
#         df_chunck.iloc[(i - 1) * n : i * n].to_csv(f"{filename}", index=False)
#     else:

#         df_chunck.iloc[(i - 1) * n : i * n].to_csv(
#             f"{filename}", mode="a", header=False, index=False
#         )


def write_csv_in_chunck(df_chunck, filename, i):
    if i == 1:
        df_chunck.to_csv(f"{filename}", index=False)
    else:

        df_chunck.to_csv(f"{filename}", mode="a", header=False, index=False)


def _get_df_top_k(
    res_df,
    take_top_k,
    metric_top_k,
    row_root=[],
    df_top_k=None,
    cols_orderTP=["tn", "fp", "fn", "tp"],
):
    res_df = pd.concat([res_df, row_root])

    from divexplorer_generalized.FP_DivergenceExplorer import (
        compute_divergence_itemsets,
    )

    res_df = compute_divergence_itemsets(
        res_df, metrics=[metric_top_k], cols_orderTP=cols_orderTP
    )
    # Drop empty set
    res_df.drop(res_df.tail(1).index, inplace=True)
    if df_top_k is None:
        # First assignment
        df_top_k = res_df
    else:
        df_top_k = pd.concat([df_top_k, res_df])
    df_top_k = df_top_k.sort_values(metric_top_k, ascending=False).iloc[0:take_top_k]
    return df_top_k


def generate_itemsets(
    generator,
    num_itemsets,
    colname_map,
    cols_orderTP=["tn", "fp", "fn", "tp"],
    attribute_id_mapping_for_compatibility=None,  ### Handling generalization/taxonomy
    save_in_progress=False,
    take_top_k=None,
    metric_top_k=None,
    row_root=None,
):

    chunck_size = 10000
    itemsets = []
    supports = []
    c1, c2, c3, c4 = [], [], [], []
    count = 0
    p = 0

    if take_top_k is not None:
        df_top_k = None
    if save_in_progress:
        import uuid

        temp_dir_name = str(uuid.uuid4())[:8]

        from pathlib import Path

        Path(temp_dir_name).mkdir(parents=True, exist_ok=True)

        import os

        filename = os.path.join(temp_dir_name, "FP_file.csv")
        print(filename)

    for sup, iset, cf_final in generator:
        ### Handling generalization/taxonomy
        compatible = True

        if compatible:
            count = count + 1
            itemsets.append(frozenset(iset))
            supports.append(sup / num_itemsets)
            c1.append(cf_final[0])
            c2.append(cf_final[1])
            c3.append(cf_final[2])
            c4.append(cf_final[3])

            if int(count / chunck_size) > p:
                p = int(count / chunck_size)
                print(p, count)
                if save_in_progress or take_top_k is not None:
                    res_df = pd.DataFrame(
                        {
                            "support": supports,
                            "itemsets": itemsets,
                            cols_orderTP[0]: c1,
                            cols_orderTP[1]: c2,
                            cols_orderTP[2]: c3,
                            cols_orderTP[3]: c4,
                        }
                    )

                    supports.clear()
                    itemsets.clear()
                    c1.clear()
                    c2.clear()
                    c3.clear()
                    c4.clear()
                    if save_in_progress:
                        if colname_map is not None:
                            res_df["itemsets"] = res_df["itemsets"].apply(
                                lambda x: [colname_map[i] for i in x]
                            )
                        write_csv_in_chunck(res_df, filename, p)

                    if take_top_k is not None:
                        # Update top k
                        if colname_map is not None:
                            res_df["itemsets"] = res_df["itemsets"].apply(
                                lambda x: frozenset([colname_map[i] for i in x])
                            )
                        df_top_k = _get_df_top_k(
                            res_df,
                            take_top_k,
                            metric_top_k,
                            row_root=row_root,
                            df_top_k=df_top_k,
                            cols_orderTP=cols_orderTP,
                        )
                    res_df = res_df[0:0]
    if save_in_progress:

        p = p + 1
        res_df = pd.DataFrame(
            {
                "support": supports,
                "itemsets": itemsets,
                cols_orderTP[0]: c1,
                cols_orderTP[1]: c2,
                cols_orderTP[2]: c3,
                cols_orderTP[3]: c4,
            }
        )
        if colname_map is not None:
            res_df["itemsets"] = res_df["itemsets"].apply(
                lambda x: [colname_map[i] for i in x]
            )
        supports.clear()
        itemsets.clear()
        c1.clear()
        c2.clear()
        c3.clear()
        c4.clear()
        write_csv_in_chunck(res_df, filename, p)
        # print("Before reading")
        import time

        s_time = time.time()
        res_df = pd.read_csv(filename)
        print(time.time() - s_time)
        s_time = time.time()
        res_df["itemsets"] = (
            res_df["itemsets"]
            .str.strip("[]")
            .apply(
                lambda x: frozenset([i.strip(" ").strip("''") for i in x.split(",")])
            )
        )
        print(time.time() - s_time)
    else:
        # print("Before pandas df")
        res_df = pd.DataFrame(
            {
                "support": supports,
                "itemsets": itemsets,
                cols_orderTP[0]: c1,
                cols_orderTP[1]: c2,
                cols_orderTP[2]: c3,
                cols_orderTP[3]: c4,
            }
        )
        if colname_map is not None:
            res_df["itemsets"] = res_df["itemsets"].apply(
                lambda x: frozenset([colname_map[i] for i in x])
            )

        if take_top_k is not None:
            # Update top k
            df_top_k = _get_df_top_k(
                res_df,
                take_top_k,
                metric_top_k,
                row_root=row_root,
                df_top_k=df_top_k,
                cols_orderTP=cols_orderTP,
            )

    if take_top_k:
        # Return all columns except the divergence and the t_value cols
        ret_cols = list(df_top_k.columns)[0:-2]
        print("EP FP", count)
        return df_top_k[ret_cols]

    return res_df


def valid_input_check(df):

    if f"{type(df)}" == "<class 'pandas.core.frame.SparseDataFrame'>":
        msg = (
            "SparseDataFrame support has been deprecated in pandas 1.0,"
            " and is no longer supported in mlxtend. "
            " Please"
            " see the pandas migration guide at"
            " https://pandas.pydata.org/pandas-docs/"
            "stable/user_guide/sparse.html#sparse-data-structures"
            " for supporting sparse data in DataFrames."
        )
        raise TypeError(msg)

    if df.size == 0:
        return
    if hasattr(df, "sparse"):
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError(
                "Due to current limitations in Pandas, "
                "if the sparse format has integer column names,"
                "names, please make sure they either start "
                "with `0` or cast them as string column names: "
                "`df.columns = [str(i) for i in df.columns`]."
            )

    # Fast path: if all columns are boolean, there is nothing to checks
    all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()
    if not all_bools:
        # Pandas is much slower than numpy, so use np.where on Numpy arrays
        if hasattr(df, "sparse"):
            if df.size == 0:
                values = df.values
            else:
                values = df.sparse.to_coo().tocoo().data
        else:
            values = df.values
        idxs = np.where((values != 1) & (values != 0))
        if len(idxs[0]) > 0:
            # idxs has 1 dimension with sparse data and 2 with dense data
            val = values[tuple(loc[0] for loc in idxs)]
            s = (
                "The allowed values for a DataFrame"
                " are True, False, 0, 1. Found value %s" % (val)
            )
            raise ValueError(s)


class FPTree(object):
    def __init__(self, rank=None):
        self.root = FPNode_CM(None)
        self.nodes = collections.defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):
        """
        Creates and returns the subtree of self conditioned on cond_item.
        Parameters
        ----------
        cond_item : int | str
            Item that the tree (self) will be conditioned on.
        minsup : int
            Minimum support threshold.
        Returns
        -------
        cond_tree : FPtree
        """
        # Find all path from root node to nodes for item
        branches = []
        count = collections.defaultdict(int)
        # cm_dict=collections.defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count
                # cm_dict[item]+=node.confusion_matrix.copy()

        # Define new ordering or deep trees may have combinatorially explosion
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        # Create conditional tree
        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted(
                [i for i in branch if i in rank], key=rank.get, reverse=True
            )
            cond_tree.insert_itemset(
                branch,
                self.nodes[cond_item][idx].count,
                cm_i=self.nodes[cond_item][idx].confusion_matrix.copy(),
            )
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert_itemset(self, itemset, count=1, cm_i=None):
        """
        Inserts a list of items into the tree.
        Parameters
        ----------
        itemset : list
            Items that will be inserted into the tree.
        count : int
            The number of occurrences of the itemset.
        """
        self.root.count += count
        self.root.confusion_matrix += cm_i

        if len(itemset) == 0:
            return

        # Follow existing path in tree as long as possible
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                child.confusion_matrix += cm_i.copy()
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = FPNode_CM(item, count, node, confusion_matrix=cm_i.copy())
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print(
            "\r%d itemset(s) from tree conditioned on items (%s)" % (count, cond_items),
            end="\n",
        )


class FPNode_CM(object):
    def __init__(self, item, count=0, parent=None, confusion_matrix=[0, 0, 0, 0]):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode_CM)
        self.confusion_matrix = confusion_matrix.copy()

        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        """Returns the top-down sequence of items from self to
        (but not including) the root node."""
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path


# mlxtend Machine Learning Library Extensions
# Author: Steve Harenberg <harenbergsd@gmail.com>
#
# License: BSD 3 clause

import math
import itertools


def fpgrowth_cm(
    df,
    cm,
    min_support=0.5,
    use_colnames=False,
    max_len=None,
    verbose=0,
    cols_orderTP=["tn", "fp", "fn", "tp"],
    attribute_id_mapping_for_compatibility=None,  ### Handling generalization/taxonomy
    save_in_progress=False,
    take_top_k=None,
    metric_top_k=None,
):
    """Get frequent itemsets from a one-hot DataFrame
    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame the encoded format. Also supports
      DataFrames with sparse data; for more info, please
      see (https://pandas.pydata.org/pandas-docs/stable/
           user_guide/sparse.html#sparse-data-structures)
      Please note that the old pandas SparseDataFrame format
      is no longer supported in mlxtend >= 0.17.2.
      The allowed values are either 0/1 or True/False.
      For example,
    ```
           Apple  Bananas   Beer  Chicken   Milk   Rice
        0   True    False   True     True  False   True
        1   True    False   True    False  False   True
        2   True    False   True    False  False  False
        3   True     True  False    False  False  False
        4  False    False   True     True   True   True
        5  False    False   True    False   True   True
        6  False    False   True    False   True  False
        7   True     True  False    False  False  False
    ```
    min_support : float (default: 0.5)
      A float between 0 and 1 for minimum support of the itemsets returned.
      The support is computed as the fraction
      transactions_where_item(s)_occur / total_transactions.
    use_colnames : bool (default: False)
      If true, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.
    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths are evaluated.
    verbose : int (default: 0)
      Shows the stages of conditional tree generation.
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
    ----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
    """
    valid_input_check(df)

    row_root = None

    if take_top_k is not None:
        # We take the first top K. For the divergence we need the measure on the overall datasrt
        row_root = dict(cm.sum())
        row_root.update({"support": 1, "itemsets": frozenset()})
        row_root = pd.DataFrame([row_root])

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )
    cm = cm.values.copy()
    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, _ = setup_fptree(df, min_support, cm)
    minsup = math.ceil(min_support * len(df.index))  # min support as count
    generator = fpg_step(
        tree,
        minsup,
        colname_map,
        max_len,
        verbose,
        attribute_id_mapping_for_compatibility=attribute_id_mapping_for_compatibility,
    )

    return generate_itemsets(
        generator,
        len(df.index),
        colname_map,
        cols_orderTP=cols_orderTP,
        attribute_id_mapping_for_compatibility=attribute_id_mapping_for_compatibility,  ### Handling generalization/taxonomy
        save_in_progress=save_in_progress,
        take_top_k=take_top_k,
        metric_top_k=metric_top_k,
        row_root=row_root,
    )


def check_compatibility_v1(iset, incompatible_items):
    compatible = True
    if incompatible_items is not None and len(iset) > 1:
        # TODO: avoid generating unless terms
        for incompatible_term_i in incompatible_items:

            cnt = 0
            for i in iset:
                if i in incompatible_term_i:
                    cnt += 1
                    if cnt > 1:
                        compatible = False

                        break

            if compatible is False:
                break

            """
            # print(iset)
            if len(set(iset).intersection(incompatible_term_i)) > 1:
                compatible = False
                break
            """

    return compatible


def check_compatibility(iset, attribute_id_mapping_for_compatibility):
    """
    Args:
        iset: current itemset to evaluate
        attribute_id_mapping_for_compatibility: dictionary item_id:attribute_name
    """

    # TODO, add check if indeed necessary
    compatible = True
    if attribute_id_mapping_for_compatibility is not None and len(iset) > 1:
        itemset_attributes = [
            attribute_id_mapping_for_compatibility[item_id] for item_id in iset
        ]
        if len(set(itemset_attributes)) != len(itemset_attributes):
            compatible = False
    return compatible


def fpg_step(
    tree,
    minsup,
    colnames,
    max_len,
    verbose,
    attribute_id_mapping_for_compatibility=None,
):
    """
    Performs a recursive step of the fpgrowth algorithm.
    Parameters
    ----------
    tree : FPTree
    minsup : int
    Yields
    ------
    lists of strings
        Set of items that has occurred in minsup itemsets.
    """
    count = 0
    items = tree.nodes.keys()
    # Move here the evaluatio of compatibility?
    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                supports_t = {i: tree.nodes[i][0].count for i in itemset}
                from operator import itemgetter

                id_min, support = min(supports_t.items(), key=itemgetter(1))
                cf_y = tree.nodes[id_min][0].confusion_matrix
                # ADDED CONDITION IMPORTANT
                #
                iset = tree.cond_items + list(itemset)
                # if check_compatibility(iset, incompatible_items):
                if check_compatibility(iset, attribute_id_mapping_for_compatibility):
                    yield support, iset, cf_y
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])

            cf_y = sum([node.confusion_matrix for node in tree.nodes[item]])

            # With generator
            # # support = sum(node.count for node in tree.nodes[item])
            # cf_y = sum(node.confusion_matrix for node in tree.nodes[item])

            # With loops
            # support = 0
            # cf_y = 0

            # for node in tree.nodes[item]:
            #     cf_y += node.confusion_matrix
            #     support += node.count
            iset = tree.cond_items + [item]
            if check_compatibility(iset, attribute_id_mapping_for_compatibility):
                yield support, iset, cf_y

    if verbose:
        tree.print_status(count, colnames)

    # Generate conditional trees to generate frequent itemsets one item larger
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)

            for sup, iset, cf_y in fpg_step(
                cond_tree,
                minsup,
                colnames,
                max_len,
                verbose,
                attribute_id_mapping_for_compatibility=attribute_id_mapping_for_compatibility,
            ):
                if check_compatibility(iset, attribute_id_mapping_for_compatibility):
                    yield sup, iset, cf_y
