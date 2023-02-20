def check_target_inputs(class_name, pred_name, target_name):
    if target_name is None and (class_name is None and pred_name is None):
        raise ValueError("Specify the target column(s)")
    if target_name is not None and (class_name is not None or pred_name is not None):
        raise ValueError(
            "Specify only a type of target: target_name if outcome target or class_name and/or pred_name for classification targets"
        )


OUTCOME = "outcome"
CLASSIFICATION = "classification"


def define_target(true_class_name, predicted_class_name, target_name):
    if (true_class_name is not None) or (predicted_class_name is not None):
        return CLASSIFICATION
    elif target_name is not None:
        return OUTCOME
    else:
        # Remove, never raised if we check before the input
        raise ValueError("None specified")


def extract_divergence_generalized(
    dfI,
    discretizations,
    generalization_dict,
    continuous_attributes,
    min_sup_divergence=0.1,
    considerOnlyContinuos=False,
    apply_generalization=False,
    true_class_name=None,  # "class",
    predicted_class_name=None,  # "predicted",
    class_map={"N": 0, "P": 1},
    target_name=None,
    FPM_type="fpgrowth",
    metrics_divergence=["d_fpr", "d_fnr", "d_accuracy", "d_error"],
    save_in_progress=False,
    preserve_interval=None,
    allow_overalp=False,
    type_experiment=None,
    verbose=False,
    keep_only_positive_divergent_items=None,
    take_top_k=None,
    metric_top_k=None,
):
    """
    keep_only_positive_divergent_items: if None, all are kept. Otherwise, keep only the one provided as input.
    """
    from copy import deepcopy

    generalization_dict_proc = deepcopy(generalization_dict)
    check_target_inputs(true_class_name, predicted_class_name, target_name)
    target_type = define_target(true_class_name, predicted_class_name, target_name)

    if type_experiment is not None:
        if verbose:
            print("E", "allow_overalp", allow_overalp)
        if type_experiment == "all_attributes":
            if verbose:
                print("E", "apply_generalization", apply_generalization)
            if apply_generalization == False:
                raise ValueError()
            if allow_overalp == False:
                raise ValueError()
    # ## Discretize the dataset

    # Discretize the dataset using the obtained discretization ranges

    from utils_discretization import discretizeDataset_from_relations

    df_s_discretized, discretized_attr = discretizeDataset_from_relations(
        dfI, discretizations, ret_original_attrs=False, allow_overalp=allow_overalp
    )

    if target_type == CLASSIFICATION:
        target_columns = [true_class_name, predicted_class_name]
    else:
        target_columns = [target_name]

    #####

    if allow_overalp:
        from utils_discretization import oneHotEncoding
        from copy import deepcopy

        df_s_discretized_discrete = deepcopy(dfI)

        attributes = list(df_s_discretized.columns.drop(target_columns))

        input_discrete_attributes = list(
            (set(attributes) - set(continuous_attributes)) - set(discretized_attr)
        )

        if input_discrete_attributes:
            df_s_discretized_discrete = oneHotEncoding(
                df_s_discretized[input_discrete_attributes]
            )

            df_s_discretized[
                df_s_discretized_discrete.columns
            ] = df_s_discretized_discrete.copy()

            attributes_df = discretized_attr + list(df_s_discretized_discrete.columns)

        else:
            attributes_df = discretized_attr

        df_s_discretized = df_s_discretized[attributes_df + target_columns]
        df_s_discretized_discrete = None
    else:
        attributes = list(df_s_discretized.columns.drop(target_columns))
        discrete_attributes = list(set(attributes) - set(continuous_attributes))

        attributes_df = discrete_attributes + discretized_attr
        df_s_discretized = df_s_discretized[attributes_df + target_columns]

    if considerOnlyContinuos:
        for k in list(generalization_dict_proc.keys()):
            if k not in continuous_attributes:
                generalization_dict_proc.pop(k, None)

    generalizations_list = None
    if allow_overalp == False:
        if apply_generalization:
            from utils_discretization import get_generalization_hierarchy

            generalizations_list = get_generalization_hierarchy(
                df_s_discretized[attributes_df], generalization_dict_proc
            )
    from divexplorer_generalized_ranking.FP_DivergenceExplorer import (
        FP_DivergenceExplorer_ranking,
    )

    fp_diver = FP_DivergenceExplorer_ranking(
        df_s_discretized,
        true_class_name=true_class_name,
        predicted_class_name=predicted_class_name,
        target_name=target_name,
        class_map=class_map,
        generalizations_obj=generalizations_list,
        preserve_interval=preserve_interval,
        already_in_one_hot_encoding=allow_overalp,
        keep_only_positive_divergent_items=keep_only_positive_divergent_items,
    )

    FP_fm_input = fp_diver.getFrequentPatternDivergence(
        min_support=min_sup_divergence,
        metrics=metrics_divergence,
        FPM_type=FPM_type,
        save_in_progress=save_in_progress,
        take_top_k=take_top_k,
        metric_top_k=metric_top_k,
    )
    return FP_fm_input
