def one_hot_encoding_attributes(data_to_encode, prefix_sep="="):
    dummy_cols = []
    for col_name, cols_data in data_to_encode.items():
        from pandas.core.arrays.categorical import factorize_from_iterable
        from pandas.core.series import Series

        _, levels = factorize_from_iterable(Series(cols_data))
        dummy_cols.extend([f"{col_name}{prefix_sep}{level}" for level in levels])
    return dummy_cols


def oneHotEncoding(dfI):
    import pandas as pd

    attributes = dfI.columns
    X_one_hot = dfI.copy()
    X_one_hot = pd.get_dummies(X_one_hot, prefix_sep="=", columns=attributes)
    X_one_hot.reset_index(drop=True, inplace=True)
    return X_one_hot


def discretizeDataset_from_relations(
    df,
    discretizations,
    suffix="_discr",
    ret_original_attrs=True,
    sep="=",
    allow_overalp=False,
):

    import operator

    ops = {
        "<": operator.lt,
        "<=": operator.le,
        "=": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }
    from copy import deepcopy

    df_s_discretized = deepcopy(df)

    discretized_attr = []

    new_values = []
    if allow_overalp:
        if allow_overalp:
            attr_name_discrete = []
            for attribute, transformations in discretizations.items():

                attr_name_discrete.append(attribute)                

                for discr_value, transformation in transformations.items():
                    item = f"{attribute}{sep}{discr_value}"
                    indexes = df.index
                    for e in range(0, len(transformation["rels"])):
                        oper, value = (
                            ops[transformation["rels"][e]],
                            transformation["vals"][e],
                        )
                        indexes = df.loc[indexes].loc[oper(df[attribute], value)].index
                    df_s_discretized[item] = 0
                    df_s_discretized.loc[indexes, item] = 1
                    new_values.append(item)
            if ret_original_attrs:
                return df_s_discretized, discretized_attr
            else:
                df_s_discretized.drop(columns=attr_name_discrete, inplace=True)
                return df_s_discretized, new_values
    else:
        attr_name_discrete = {}
        for attribute, transformations in discretizations.items():
            discretized_name = f"{attribute}{suffix}"
            if discretized_name not in df_s_discretized:
                df_s_discretized[discretized_name] = df_s_discretized[attribute]
                discretized_attr.append(discretized_name)
                attr_name_discrete[discretized_name] = attribute

            for discr_value, transformation in transformations.items():
                indexes = df.index
                for e in range(0, len(transformation["rels"])):
                    oper, value = (
                        ops[transformation["rels"][e]],
                        transformation["vals"][e],
                    )
                    indexes = df.loc[indexes].loc[oper(df[attribute], value)].index
                df_s_discretized.loc[indexes, discretized_name] = discr_value
                new_values.append(discr_value)

    if ret_original_attrs:
        return df_s_discretized, discretized_attr
    else:
        df_s_discretized.drop(columns=attr_name_discrete.values(), inplace=True)
        df_s_discretized.rename(columns=attr_name_discrete, inplace=True)
        return df_s_discretized, list(attr_name_discrete.values())


def get_generalization_hierarchy(df_encode, generalization_dict):
    # from utils_discretization import one_hot_encoding_attributes

    attributes_one_hot = one_hot_encoding_attributes(df_encode)

    # # Incompatible attribute values

    # Incompatible attribute values
    # The items obtained from the same attribute are incompatible
    from utils_hierarchy import incompatible_attribute_value

    shared_attributes_incompatible = incompatible_attribute_value(attributes_one_hot)
    # # Generalizations

    # Receive a dictionary of generalizations (as the one produced by the tree) and store in an object this information

    from generalized_attributes_hierarchy import Generalizations_hierarchy

    counter_id = len(attributes_one_hot)
    generalizations_list = Generalizations_hierarchy(
        counter_id, shared_attributes_incompatible
    )

    generalizations_list.add_generalizations(
        generalization_dict, attributes_one_hot  # , level_struct=False
    )

    return generalizations_list
