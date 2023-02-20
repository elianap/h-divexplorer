from mlxtend.frequent_patterns.apriori import fpc


def apriori_divergence_generalization_hierarchy_old(
    df,
    df_true_pred,
    min_support=0.5,
    use_colnames=False,
    max_len=None,
    verbose=0,
    low_memory=False,
    cols_orderTP=["tn", "fp", "fn", "tp"],
    sortedV="support",
    generalization=None,
    incompatible_items={},
    enable_cprofile=False,
):
    if enable_cprofile:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
    from copy import deepcopy
    import numpy as np
    import pandas as pd

    df_true_pred_gen = deepcopy(df_true_pred)
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

    def filterColumns_only_gen(df_filter, gen_cols):

        indexes = df_filter[(df_filter[df_filter.columns[list(gen_cols)]] > 0)].any(1)
        return indexes

    def filterColumns_gen(df_filter, gen_cols, attr_cols):
        indexes = filterColumns_only_gen(df_filter, gen_cols)
        return filterColumns(df_true_pred[indexes], attr_cols)

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

    def _support_count(_x, _n_rows):
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
        out = np.sum(_x, axis=0)
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

    name_mapping = generalization.name_mapping_flatten()
    if verbose:
        print("-----------------------------------------")
        print(name_mapping)
        print("-----------------------------------------")
    name_mapping_new_columns = {}

    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    def generate_new_combinations_except(old_combinations, incompatible):
        """
        Generator of all combinations based on the last state of Apriori algorithm
        Parameters
        -----------
        old_combinations: np.array
            All combinations with enough support in the last step
            Combinations are represented by a matrix.
            Number of columns is equal to the combination size
            of the previous step.
            Each row represents one combination
            and contains item type ids in the ascending order
            ```
                0        1
            0      15       20
            1      15       22
            2      17       19
            ```

        Returns
        -----------
        Generator of all combinations from the last step x items
        from the previous step.

        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

        """

        verbose = False
        list_no = []
        items_types_in_previous_step = np.unique(old_combinations.flatten())
        if verbose:
            print(
                "----------------------------------------------------------------------------------------------------------------------"
            )
            print(items_types_in_previous_step)
        for old_combination in old_combinations:
            max_combination = old_combination[-1]
            mask = items_types_in_previous_step > max_combination
            valid_items = items_types_in_previous_step[mask]
            old_tuple = tuple(old_combination)
            if verbose:
                # print("max_combination", max_combination)
                # print(mask)
                print("valid_items", valid_items)
                print("old_tuple", old_tuple)
            for item in valid_items:
                incomp = False
                if verbose:
                    print("item", item)
                    print("old_tuple", old_tuple)
                    print(len(old_tuple))

                if (
                    len(old_tuple)
                    == 1
                    # and (old_tuple[0] in incompatible)
                    # and (item in incompatible[old_tuple[0]])
                ):
                    list_incomp = [
                        e
                        for e, incompatible_i in enumerate(incompatible)
                        if old_tuple[0] in incompatible_i and item in incompatible_i
                    ]
                    if list_incomp != []:
                        if verbose:
                            print(incompatible, list_incomp)
                            print(
                                list_incomp,
                                [incompatible[i] for i in list_incomp],
                                incompatible,
                                old_tuple[0],
                            )

                            print("\n\nINCOMPATIBLE")
                            print(item)
                            print(old_tuple[0])
                            print(
                                f"------------------------------------------------------------------------->incompatible {old_tuple} {item}"
                            )
                        # continue
                        list_no.append((old_tuple, item))
                        continue
                elif len(old_tuple) > 1:
                    for value in old_tuple:
                        list_incomp = [
                            e
                            for e, incompatible_i in enumerate(incompatible)
                            if value in incompatible_i and item in incompatible_i
                        ]
                        if list_incomp != []:
                            if verbose:
                                print(
                                    old_tuple,
                                    value,
                                    item,
                                    list_incomp,
                                    [incompatible[i] for i in list_incomp],
                                )
                                print(
                                    "------------------------------------------------------------------------->incompatible {old_tuple} {item}"
                                )
                            incomp = True
                if incomp:
                    continue
                if verbose:
                    print(
                        f"-------------------------------------->OK {old_tuple} {item}"
                    )

                yield from old_tuple
                yield item
        if verbose:
            print(incompatible)
            print(list_no)

    incompatible_items_list = list(incompatible_items.values())

    while max_itemset and max_itemset < (max_len or float("inf")):
        next_max_itemset = max_itemset + 1
        if verbose:
            print("\n\n--------------------------")

        combin_all = generate_new_combinations_except(
            itemset_dict[max_itemset], incompatible_items_list
        )
        combin_all = np.fromiter(combin_all, dtype=int)
        combin_all = combin_all.reshape(-1, next_max_itemset)
        if verbose:
            print(combin_all)
        combin = combin_all
        X = df_true_pred_gen.values
        if combin.size == 0:
            break
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
                    sum_values(filterColumns(df_true_pred_gen, itemset)[cols_orderTP])
                    for itemset in itemset_dict[next_max_itemset]
                ]
            )
            max_itemset = next_max_itemset
        else:
            # Exit condition
            break

    all_res = []
    if verbose:
        print(conf_metrics)
        print(support_dict)
        print(itemset_dict)
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")
        conf_metrics_cols = pd.DataFrame(list(conf_metrics[k]), columns=cols_orderTP)

        res = pd.concat((support, itemsets, conf_metrics_cols), axis=1)
        all_res.append(res)
    if verbose:
        print(all_res)

    res_df = pd.concat(all_res)
    res_df.columns = ["support", "itemsets"] + cols_orderTP

    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        mapping.update(name_mapping_new_columns)

        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )

    res_df["length"] = res_df["itemsets"].str.len()
    res_df["support_count"] = np.sum(res_df[cols_orderTP], axis=1)
    from utils_metrics_FPx import fpr_df, fnr_df, accuracy_df

    res_df["fpr"] = fpr_df(res_df[cols_orderTP])
    res_df["fnr"] = fnr_df(res_df[cols_orderTP])
    res_df["accuracy"] = accuracy_df(res_df[cols_orderTP])

    res_df.sort_values(sortedV, ascending=False, inplace=True)
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used
    if enable_cprofile:
        pr.disable()
        pr.print_stats(sort="time")
    return res_df


def apriori_divergence_generalization_hierarchy_ok(
    df,
    df_true_pred,
    min_support=0.5,
    use_colnames=False,
    max_len=None,
    verbose=0,
    low_memory=False,
    cols_orderTP=["tn", "fp", "fn", "tp"],
    sortedV="support",
    generalization=None,
    incompatible_items={},
    enable_cprofile=False,
):
    from copy import deepcopy
    import numpy as np
    import pandas as pd

    if enable_cprofile:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()

    df_true_pred_gen = deepcopy(df_true_pred)

    ## Redundant
    df_true_pred_values = df_true_pred_gen.values
    df_cm_values = df_true_pred_gen[cols_orderTP].values

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

    # def filterColumns(df_filter, cols):
    #     return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]

    # def filterColumns_only_gen(df_filter, gen_cols):

    #     indexes = df_filter[(df_filter[df_filter.columns[list(gen_cols)]] > 0)].any(1)
    #     return indexes

    # def filterColumns_gen(df_filter, gen_cols, attr_cols):
    #     indexes = filterColumns_only_gen(df_filter, gen_cols)
    #     return filterColumns(df_true_pred[indexes], attr_cols)

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

    def _support_count(_x, _n_rows):
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
        out = np.sum(_x, axis=0)
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

    # conf_metrics = {
    #     0: np.asarray([sum_values(df_true_pred[cols_orderTP])]),
    #     1: np.asarray(
    #         [
    #             sum_values(filterColumns(df_true_pred, item)[cols_orderTP])
    #             for item in itemset_dict[1]
    #         ]
    #     ),
    # }

    conf_metrics = {
        0: np.asarray([sum_values(df_true_pred[cols_orderTP])]),
        # Get columns where all fields are equal to 1
        # Get column matric with [:-4]
        1: np.asarray(
            [
                sum_values(
                    df_cm_values[df_true_pred_values[:, list(item)].all(1)][:, -4:]
                )
                for item in itemset_dict[1]
            ]
        ),
    }

    name_mapping = generalization.name_mapping_flatten()

    name_mapping_new_columns = {}

    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    def generate_new_combinations_except(old_combinations, incompatible):
        """
        Generator of all combinations based on the last state of Apriori algorithm
        Parameters
        -----------
        old_combinations: np.array
            All combinations with enough support in the last step
            Combinations are represented by a matrix.
            Number of columns is equal to the combination size
            of the previous step.
            Each row represents one combination
            and contains item type ids in the ascending order
            ```
                0        1
            0      15       20
            1      15       22
            2      17       19
            ```

        Returns
        -----------
        Generator of all combinations from the last step x items
        from the previous step.

        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

        """

        # verbose = False
        list_no = []
        items_types_in_previous_step = np.unique(old_combinations.flatten())

        for old_combination in old_combinations:
            max_combination = old_combination[-1]
            mask = items_types_in_previous_step > max_combination
            valid_items = items_types_in_previous_step[mask]
            old_tuple = tuple(old_combination)
            if verbose:
                print("valid_items", valid_items)
                print("old_tuple", old_tuple)
            for item in valid_items:
                compatible = True

                if verbose:
                    print("~~~~~~~~", old_tuple, item)
                    print(frozenset(old_tuple))
                    print(frozenset(old_tuple).union([item]))

                iset = frozenset(old_tuple).union([item])

                if incompatible is not None and len(iset) > 1:
                    for incompatible_term_i in incompatible:
                        if len(set(iset).intersection(incompatible_term_i)) > 1:
                            compatible = False
                            if verbose:
                                print(
                                    "------------------> INCOMPATIBLE",
                                    iset,
                                    incompatible_term_i,
                                )
                            break
                if compatible:
                    if verbose:
                        print("N------------------> OK", iset, incompatible_term_i)
                    yield from old_tuple
                    yield item
                else:
                    continue

    incompatible_items_list = list(incompatible_items.values())

    while max_itemset and max_itemset < (max_len or float("inf")):
        next_max_itemset = max_itemset + 1
        if verbose:
            print("\n\n--------------------------")

        combin_all = generate_new_combinations_except(
            itemset_dict[max_itemset], incompatible_items_list
        )
        combin_all = np.fromiter(combin_all, dtype=int)
        combin_all = combin_all.reshape(-1, next_max_itemset)

        if verbose:
            print(combin_all)
        combin = combin_all
        X = df_true_pred_gen.values
        if combin.size == 0:
            break
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

            # conf_metrics[next_max_itemset] = np.asarray(
            #     [
            #         sum_values(filterColumns(df_true_pred_gen, itemset)[cols_orderTP])
            #         for itemset in itemset_dict[next_max_itemset]
            #     ]
            # )

            # TODO --> use _mask
            conf_metrics[next_max_itemset] = np.asarray(
                [
                    sum_values(
                        df_cm_values[df_true_pred_values[:, list(itemset)].all(1)][
                            :, -4:
                        ]
                    )
                    for itemset in itemset_dict[next_max_itemset]
                ]
            )

            max_itemset = next_max_itemset
        else:
            # Exit condition
            break

    all_res = []
    if verbose:
        print(conf_metrics)
        print(support_dict)
        print(itemset_dict)

    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")
        conf_metrics_cols = pd.DataFrame(list(conf_metrics[k]), columns=cols_orderTP)

        res = pd.concat((support, itemsets, conf_metrics_cols), axis=1)
        all_res.append(res)
    if verbose:
        print(all_res)

    res_df = pd.concat(all_res)
    res_df.columns = ["support", "itemsets"] + cols_orderTP

    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        mapping.update(name_mapping_new_columns)

        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )

    res_df["length"] = res_df["itemsets"].str.len()
    res_df["support_count"] = np.sum(res_df[cols_orderTP], axis=1)
    from utils_metrics_FPx import fpr_df, fnr_df, accuracy_df

    res_df["fpr"] = fpr_df(res_df[cols_orderTP])
    res_df["fnr"] = fnr_df(res_df[cols_orderTP])
    res_df["accuracy"] = accuracy_df(res_df[cols_orderTP])

    res_df.sort_values(sortedV, ascending=False, inplace=True)
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used
    if enable_cprofile:
        pr.disable()
        pr.print_stats(sort="time")
    return res_df


def convert_in_pandas(
    support_dict_i, itemset_dict_i, conf_metrics_i, cols_orderTP, frozenset_form=True
):
    import pandas as pd

    support = pd.Series(support_dict_i)
    if frozenset_form:
        itemsets = pd.Series([frozenset(i) for i in itemset_dict_i], dtype="object")
    else:
        itemsets = pd.Series([i for i in itemset_dict_i], dtype="object")
    conf_metrics_cols = pd.DataFrame(list(conf_metrics_i), columns=cols_orderTP)
    res = pd.concat((support, itemsets, conf_metrics_cols), axis=1)
    return res


def add_column_names(res_df, columns_names):
    mapping = {idx: item for idx, item in enumerate(columns_names)}
    # mapping.update(name_mapping_new_columns)

    res_df["itemsets"] = res_df["itemsets"].apply(
        lambda x: frozenset([mapping[i] for i in x])
    )
    return res_df


def add_column_names_array(res_array, columns_names):
    import numpy as np

    array = res_array[:, 1]
    mapping = {idx: item for idx, item in enumerate(columns_names)}
    n = np.asarray([frozenset([mapping[i] for i in x]) for x in array])
    return np.hstack((res_array, n.reshape(-1, 1)))


def add_info_divergence(res_df, cols_orderTP):
    import numpy as np

    res_df["length"] = res_df["itemsets"].str.len()
    res_df["support_count"] = np.sum(res_df[cols_orderTP], axis=1)
    from utils_metrics_FPx import fpr_df, fnr_df, accuracy_df

    res_df["fpr"] = fpr_df(res_df[cols_orderTP])
    res_df["fnr"] = fnr_df(res_df[cols_orderTP])
    res_df["accuracy"] = accuracy_df(res_df[cols_orderTP])
    return res_df


def add_info_divergence_array(res_array, cols_orderTP):
    import numpy as np

    array = res_array[:, -4:]

    l_ar = [
        i.shape[0] if type(i) != tuple else len(i) for i in res_array[:, 1]
    ]  # res_df["itemsets"].str.len()
    s_ar = np.sum(array, axis=1)
    res = np.vstack(
        (
            l_ar,
            s_ar,
            fpr_arr(array, cols_orderTP),
            fnr_arr(array, cols_orderTP),
            accuracy_arr(array, cols_orderTP),
        )
    )
    res = np.hstack((res_array, res.T))
    return res


def get_fraction(array, id_num, id_dem):
    import numpy as np

    a = np.sum(array[:, id_num], axis=1)
    b = np.sum(array[:, id_dem], axis=1)
    return np.divide(a, b, where=b != 0.0, out=np.zeros_like(b))


def fnr_arr(array, cols_orderTP):
    id_num = [cols_orderTP.index("fn")]
    id_dem = [cols_orderTP.index("fn"), cols_orderTP.index("tp")]
    return get_fraction(array, id_num, id_dem)


def accuracy_arr(array, cols_orderTP):
    id_num = [cols_orderTP.index("tp"), cols_orderTP.index("tn")]
    id_dem = list(range(0, 4))
    return get_fraction(array, id_num, id_dem)


def fpr_arr(array, cols_orderTP):
    id_num = [cols_orderTP.index("fp")]
    id_dem = [cols_orderTP.index("fp"), cols_orderTP.index("tn")]
    return get_fraction(array, id_num, id_dem)


def apriori_divergence_generalization_hierarchy(
    df,
    df_true_pred,
    min_support=0.5,
    use_colnames=False,
    max_len=None,
    verbose=0,
    low_memory=False,
    cols_orderTP=["tn", "fp", "fn", "tp"],
    sortedV="support",
    #generalization=None,
    incompatible_items={},
    enable_cprofile=False,
    save_in_progress=False,
):
    from copy import deepcopy
    import numpy as np
    import pandas as pd

    if enable_cprofile:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()

    df_true_pred_gen = deepcopy(df_true_pred)

    ## Redundant
    df_true_pred_values = df_true_pred_gen.values
    df_cm_values = df_true_pred_gen[cols_orderTP].values

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

    # def filterColumns(df_filter, cols):
    #     return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]

    # def filterColumns_only_gen(df_filter, gen_cols):

    #     indexes = df_filter[(df_filter[df_filter.columns[list(gen_cols)]] > 0)].any(1)
    #     return indexes

    # def filterColumns_gen(df_filter, gen_cols, attr_cols):
    #     indexes = filterColumns_only_gen(df_filter, gen_cols)
    #     return filterColumns(df_true_pred[indexes], attr_cols)

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

    def _support_count(_x, _n_rows):
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
        out = np.sum(_x, axis=0)
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

    # conf_metrics = {
    #     0: np.asarray([sum_values(df_true_pred[cols_orderTP])]),
    #     1: np.asarray(
    #         [
    #             sum_values(filterColumns(df_true_pred, item)[cols_orderTP])
    #             for item in itemset_dict[1]
    #         ]
    #     ),
    # }

    conf_metrics = {
        0: np.asarray([sum_values(df_true_pred[cols_orderTP])]),
        # Get columns where all fields are equal to 1
        # Get column matric with [:-4]
        1: np.asarray(
            [
                sum_values(
                    df_cm_values[df_true_pred_values[:, list(item)].all(1)][:, -4:]
                )
                for item in itemset_dict[1]
            ]
        ),
    }

    if save_in_progress:

        import uuid

        temp_dir_name = str(uuid.uuid4())[:8]

        from pathlib import Path

        Path(temp_dir_name).mkdir(parents=True, exist_ok=True)

        import os

        filename = os.path.join(temp_dir_name, "fp_file")
        print(filename)
        for k in sorted(itemset_dict):
            print(len(itemset_dict[k]))
            res = convert_in_pandas(
                support_dict[k],
                itemset_dict[k],
                conf_metrics[k],
                cols_orderTP,
                frozenset_form=False,
            )

            res_values = add_info_divergence_array(res.values, cols_orderTP)
            res_values = add_column_names_array(res_values, df.columns)

            if k == 0:
                with open(f"{filename}.npy", "wb") as f:
                    np.save(f, res_values, allow_pickle=True)
            else:
                with open(f"{filename}.npy", "ab") as f:
                    np.save(f, res_values, allow_pickle=True)
        itemset_dict.pop(0)
        support_dict, conf_metrics = {}, {}

    # name_mapping = generalization.name_mapping_flatten()

    # name_mapping_new_columns = {}

    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    def generate_new_combinations_except(old_combinations, incompatible):
        """
        Generatsor of all combinations based on the last state of Apriori algorithm
        Parameters
        -----------
        old_combinations: np.array
            All combinations with enough support in the last step
            Combinations are represented by a matrix.
            Number of columns is equal to the combination size
            of the previous step.
            Each row represents one combination
            and contains item type ids in the ascending order
            ```
                0        1
            0      15       20
            1      15       22
            2      17       19
            ```

        Returns
        -----------
        Generator of all combinations from the last step x items
        from the previous step.

        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

        """

        # verbose = False
        list_no = []
        items_types_in_previous_step = np.unique(old_combinations.flatten())

        for old_combination in old_combinations:
            max_combination = old_combination[-1]
            mask = items_types_in_previous_step > max_combination
            valid_items = items_types_in_previous_step[mask]
            old_tuple = tuple(old_combination)
            if verbose:
                print("valid_items", valid_items)
                print("old_tuple", old_tuple)
            for item in valid_items:
                compatible = True

                if verbose:
                    print("~~~~~~~~", old_tuple, item)
                    print(frozenset(old_tuple))
                    print(frozenset(old_tuple).union([item]))

                iset = frozenset(old_tuple).union([item])

                if incompatible is not None and len(iset) > 1:
                    for incompatible_term_i in incompatible:
                        if len(set(iset).intersection(incompatible_term_i)) > 1:
                            compatible = False
                            if verbose:
                                print(
                                    "------------------> INCOMPATIBLE",
                                    iset,
                                    incompatible_term_i,
                                )
                            break
                if compatible:
                    if verbose:
                        print("N------------------> OK", iset, incompatible_term_i)
                    yield from old_tuple
                    yield item
                else:
                    continue

    incompatible_items_list = list(incompatible_items.values())

    while max_itemset and max_itemset < (max_len or float("inf")):
        next_max_itemset = max_itemset + 1
        if verbose:
            print("\n\n--------------------------")
        # TODO: as size increases, iterate over combination in batches
        combin_all = generate_new_combinations_except(
            itemset_dict[max_itemset], incompatible_items_list
        )
        combin_all = np.fromiter(combin_all, dtype=int)
        combin_all = combin_all.reshape(-1, next_max_itemset)

        if verbose:
            print(combin_all)
        combin = combin_all
        X = df_true_pred_gen.values
        if combin.size == 0:
            break
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
            # print(next_max_itemset, len(itemset_dict[next_max_itemset]))
            support_dict[next_max_itemset] = np.array(support[_mask])

            # conf_metrics[next_max_itemset] = np.asarray(
            #     [
            #         sum_values(filterColumns(df_true_pred_gen, itemset)[cols_orderTP])
            #         for itemset in itemset_dict[next_max_itemset]
            #     ]
            # )

            # TODO --> use _mask
            conf_metrics[next_max_itemset] = np.asarray(
                [
                    sum_values(
                        df_cm_values[df_true_pred_values[:, list(itemset)].all(1)][
                            :, -4:
                        ]
                    )
                    for itemset in itemset_dict[next_max_itemset]
                ]
            )

            if save_in_progress:
                print(len(itemset_dict[next_max_itemset]))
                res = convert_in_pandas(
                    support_dict[next_max_itemset],
                    itemset_dict[next_max_itemset],
                    conf_metrics[next_max_itemset],
                    cols_orderTP,
                    frozenset_form=False,
                )

                res_values = add_info_divergence_array(res.values, cols_orderTP)
                res_values = add_column_names_array(res_values, df.columns)

                with open(f"{filename}.npy", "ab") as f:
                    np.save(f, res_values, allow_pickle=True)
                itemset_dict.pop(next_max_itemset - 1)
                support_dict, conf_metrics = {}, {}

            max_itemset = next_max_itemset
        else:
            # Exit condition
            break

    if verbose:
        print(conf_metrics)
        print(support_dict)
        print(itemset_dict)

    if save_in_progress == False:
        all_res = []
        for k in sorted(itemset_dict):
            res = convert_in_pandas(
                support_dict[k], itemset_dict[k], conf_metrics[k], cols_orderTP
            )
            all_res.append(res)

        if verbose:
            print(all_res)

        res_df = pd.concat(all_res)
        res_df.columns = ["support", "itemsets"] + cols_orderTP

        if use_colnames:
            res_df = add_column_names(res_df, df.columns)  # , name_mapping_new_columns)

        res_df = add_info_divergence(res_df, cols_orderTP)
        res_df.sort_values(sortedV, ascending=False, inplace=True)
        res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used
    if enable_cprofile:
        pr.disable()
        pr.print_stats(sort="time")
    return res_df if save_in_progress == False else (filename, max_itemset)
