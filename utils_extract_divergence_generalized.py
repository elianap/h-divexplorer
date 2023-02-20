from divexplorer_generalized.utils_metrics_FPx import false_discovery_rate_df


def extract_divergence_generalized(
    dfI,
    discretizations,
    generalization_dict,
    continuous_attributes,
    min_sup_divergence=0.1,
    considerOnlyContinuos=False,
    apply_generalization=False,
    true_class_name="class",
    predicted_class_name="predicted",
    class_map={"N": 0, "P": 1},
    FPM_type="fpgrowth",
    metrics_divergence=["d_fpr", "d_fnr", "d_accuracy", "d_error"],
    save_in_progress=False,
    preserve_interval=None,
    allow_overalp=False,
    type_experiment=None,
    verbose=False,
    keep_only_positive_divergent_items=None,
    take_top_k = None,
    metric_top_k = None
):
    """
    keep_only_positive_divergent_items: if None, all are kept. Otherwise, keep only the one provided as input.
    """
    from copy import deepcopy

    generalization_dict_proc = deepcopy(generalization_dict)

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

    # In the case of allow overlapping, the continuous attribute in the discretization
    # tree are already discretized
    from utils_discretization import discretizeDataset_from_relations

    df_s_discretized, discretized_attr = discretizeDataset_from_relations(
        dfI, discretizations, ret_original_attrs=False, allow_overalp=allow_overalp
    )

    #####

    if allow_overalp:
        from utils_discretization import oneHotEncoding
        from copy import deepcopy

        df_s_discretized_discrete = deepcopy(dfI)

        attributes = list(
            df_s_discretized.columns.drop([true_class_name, predicted_class_name])
        )

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

        df_s_discretized = df_s_discretized[
            attributes_df + [true_class_name, predicted_class_name]
        ]
        df_s_discretized_discrete = None
    else:
        attributes = list(
            df_s_discretized.columns.drop([true_class_name, predicted_class_name])
        )
        discrete_attributes = list(set(attributes) - set(continuous_attributes))

        attributes_df = discrete_attributes + discretized_attr
        df_s_discretized = df_s_discretized[
            attributes_df + [true_class_name, predicted_class_name]
        ]

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

    from divexplorer_generalized.FP_DivergenceExplorer import FP_DivergenceExplorer

    fp_diver = FP_DivergenceExplorer(
        df_s_discretized,
        true_class_name,
        predicted_class_name,
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
        take_top_k = take_top_k,
        metric_top_k = metric_top_k
    )
    return FP_fm_input
