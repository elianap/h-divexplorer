def incompatible_attribute_value(columns_one_hot):
    id_attribute_map = {e: c.split("=")[0] for e, c in enumerate(columns_one_hot)}
    shared_attributes_incompatible = {}
    for k, v in id_attribute_map.items():
        if v not in shared_attributes_incompatible:
            shared_attributes_incompatible[v] = []
        shared_attributes_incompatible[v].append(k)
    return {k: frozenset(v) for k, v in shared_attributes_incompatible.items()}


def numberPresent(list_i, value):
    return len([j for j in list_i if j == value])


def check_in_levels(elements):
    return all([type(i) == int for i in elements])


def convertGeneralizationInLevels(generalization):
    generalization_sorted_level = {}
    for attr, gens in generalization.items():
        keys_v = list(gens.keys())
        # TODO
        # Simple test, check if keys are int and dict keys
        if check_in_levels(keys_v) and type(gens[keys_v[0]]) == dict:
            # If the dictionary has a level structure, assume it is correct
            # TODO check
            return generalization
        generalization_sorted_level[attr] = {}
        i = 0
        while keys_v:
            find = False
            # i = 0
            if i >= len(keys_v):
                raise ValueError(f"Not able to find in the list {keys_v} ({i})- TODO")
            v = keys_v[i]
            gen_v = gens[v]
            if v not in gens.values():
                level = 1
                if level not in generalization_sorted_level[attr]:
                    generalization_sorted_level[attr][level] = {}
                generalization_sorted_level[attr][level][v] = gen_v
                keys_v.pop(i)
                i = 0
            else:
                levels = generalization_sorted_level[attr].keys()
                count_presence = 0
                for level, gen_level in generalization_sorted_level[attr].items():
                    count_presence = count_presence + numberPresent(
                        gen_level.values(), v
                    )
                    if count_presence == numberPresent(gens.values(), v):
                        if level + 1 not in generalization_sorted_level[attr]:
                            generalization_sorted_level[attr][level + 1] = {}
                        generalization_sorted_level[attr][level + 1][v] = gen_v
                        keys_v.pop(i)
                        find = True
                        i = 0
                        break
                if find == False:
                    i = i + 1
                    continue
    return generalization_sorted_level


def extend_dataset_with_hierarchy(
    df_one_hot, generalizations_obj, drop_intervals=None, sep="=", verbose=False
):
    # Dataset in one hot encoding
    from copy import deepcopy

    X_one_hot_extend = deepcopy(df_one_hot)
    for (
        (id_new_col, name_new_col),
        id_cols,
    ) in generalizations_obj.dict_name_id_generalizations_flatten().items():
        if verbose:
            print(id_new_col, name_new_col, id_cols)

        
        X_one_hot_extend[name_new_col] = (
            X_one_hot_extend[X_one_hot_extend.columns[list(id_cols)]].any(axis = 1).astype(int)
        )
        if X_one_hot_extend.shape[1] - 1 != id_new_col:
            raise ValueError("Not correspond id and name")
            # Sol: update the value

    new_remapped_columns = None
    if drop_intervals:
        enumerate_cols = dict(enumerate(X_one_hot_extend.columns))
        # current_enumerate_cols = dict(enumerate(X_one_hot_extend_current.columns))
        from copy import deepcopy

        X_one_hot_extend_current = deepcopy(X_one_hot_extend)
        for attribute_d, vals in drop_intervals.items():
            drop = [f"{attribute_d}{sep}{v[0]}" for v in vals]
            X_one_hot_extend_current = X_one_hot_extend_current.drop(columns=drop)

        current_enumerate_cols = dict(enumerate(X_one_hot_extend_current.columns))
        current_enumerate_cols_reversed = {
            v: k for k, v in current_enumerate_cols.items()
        }
        new_remapped_columns = {
            k: current_enumerate_cols_reversed[v]
            for k, v in enumerate_cols.items()
            if v in current_enumerate_cols_reversed
        }

        return X_one_hot_extend_current, new_remapped_columns
    else:
        return X_one_hot_extend, new_remapped_columns
