def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def getOverlap_normalized(a, b):
    n = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    d1 = b[1] - b[0]
    d2 = a[1] - a[0]
    return n / max(d1, d2)


def get_overlap_intervals(interval_1, interval_2):

    # |-----------------|
    #        |-----------------|
    #        |----------|         Overlap
    left_edge_overlap = max(interval_1[0], interval_2[0])
    right_edge_overlap = min(interval_1[1], interval_2[1])

    if left_edge_overlap > right_edge_overlap:
        # |---|
        #        |-----------------|
        #
        return 0
    return right_edge_overlap - left_edge_overlap


def get_recall_interval(true_interval, pred_interval):
    overlap = get_overlap_intervals(true_interval, pred_interval)
    return overlap / (true_interval[1] - true_interval[0])


def get_precision_interval(true_interval, pred_interval):
    overlap = get_overlap_intervals(true_interval, pred_interval)
    width = pred_interval[1] - pred_interval[0]

    if pred_interval[1] == pred_interval[0]:
        if pred_interval[1] > true_interval[1]:
            return 0
        else:
            print("One point overlap", true_interval, pred_interval)
            return 0

    return overlap / width


def get_f_measure_interval(true_interval, pred_interval):
    precision = get_precision_interval(true_interval, pred_interval)
    if precision == 0:
        return 0
    recall = get_recall_interval(true_interval, pred_interval)
    if recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def convert_to_interval(rel_vals, min_v, max_v):
    interval = [min_v, max_v]
    for rel, val in rel_vals:
        if (rel == "<") or (rel == "<="):
            interval[1] = val
        elif (rel == ">") or (rel == ">="):
            interval[0] = val
        elif rel == "=":
            interval[0] = val
            interval[1] = val
        else:
            raise ValueError(rel)
    return interval


def check_interval_similarity_with_bias(
    injected_biases, discretization_intervals, min_max_values
):

    highest_f_measure = {}
    f_measure_dict = {}

    # For each attribute is in the injected pattern
    for attribute in injected_biases:

        f_measure_dict[attribute] = {}
        highest_f_measure[attribute] = {}

        # Check if the attribute has been discretized.
        if attribute in discretization_intervals:

            # If yes, we evaluate the overlap

            # Min and max values of the attribute
            min_v, max_v = min_max_values[attribute]

            # For each interval of the attribute where we have an injected bias
            for bias_i in injected_biases[attribute]:

                bias_i_fr = frozenset(bias_i)

                f_measure_dict[attribute][bias_i_fr] = {}

                # We convert the bias to interval
                # e.g.
                # a: [('>=', 0.7), ('<=', 1)] --> [0.7, 1]
                # b: [('>=', 0.5)]  -->  [0.5, 1.0]
                bias_i_interval = convert_to_interval(bias_i, min_v, max_v)

                # For each discretiation interval of the attribute.
                # e.g.
                # a: [ [0.0, 0.69], [0.7, 1.0] ]
                # b: [ [0.0, 0.49], [0.5, 1.0] ]

                for discr_i_interval in discretization_intervals[attribute]:

                    # Get f-measure interval
                    f_measure = get_f_measure_interval(
                        bias_i_interval, discr_i_interval
                    )

                    # overlap_value_normalized = getOverlap_normalized(
                    #     bias_i_interval, discr_i_interval
                    # )

                    # Keep track of all results for each discretizaion interval
                    f_measure_dict[attribute][bias_i_fr][
                        "-".join(list(map(str, discr_i_interval)))
                    ] = f_measure

                    ### Update max value
                    if bias_i_fr not in highest_f_measure[attribute]:
                        highest_f_measure[attribute][bias_i_fr] = (
                            discr_i_interval,
                            f_measure,
                        )
                    else:
                        if f_measure > highest_f_measure[attribute][bias_i_fr][1]:
                            # Update best matching interval and highest_f_measure
                            highest_f_measure[attribute][bias_i_fr] = (
                                discr_i_interval,
                                f_measure,
                            )
        # If the attribute is not discretized, by defaul the overalap is 0

    return highest_f_measure, f_measure_dict


def check_interval_similarity_with_bias_test(
    injected_biases, discretization_intervals, min_max_values
):

    highest_f_measure = {}
    f_measure_dict = {}

    # For each attribute is in the injected pattern
    for attribute in injected_biases:

        f_measure_dict[attribute] = {}
        highest_f_measure[attribute] = {}

        print(attribute, discretization_intervals)

        # Check if the attribute has been discretized.
        if attribute in discretization_intervals:

            # If yes, we evaluate the overlap

            # Min and max values of the attribute
            min_v, max_v = min_max_values[attribute]

            # For each interval of the attribute where we have an injected bias
            for bias_i in injected_biases[attribute]:
                print(bias_i)

                bias_i_fr = frozenset(bias_i)

                f_measure_dict[attribute][bias_i_fr] = {}

                # We convert the bias to interval
                # e.g.
                # a: [('>=', 0.7), ('<=', 1)] --> [0.7, 1]
                # b: [('>=', 0.5)]  -->  [0.5, 1.0]
                bias_i_interval = convert_to_interval(bias_i, min_v, max_v)

                # For each discretiation interval of the attribute.
                # e.g.
                # a: [ [0.0, 0.69], [0.7, 1.0] ]
                # b: [ [0.0, 0.49], [0.5, 1.0] ]

                for discr_i_interval in discretization_intervals[attribute]:

                    # Get f-measure interval
                    f_measure = get_f_measure_interval(
                        bias_i_interval, discr_i_interval
                    )

                    # overlap_value_normalized = getOverlap_normalized(
                    #     bias_i_interval, discr_i_interval
                    # )

                    # Keep track of all results for each discretizaion interval
                    f_measure_dict[attribute][bias_i_fr][
                        "-".join(list(map(str, discr_i_interval)))
                    ] = f_measure

                    ### Update max value
                    if bias_i_fr not in highest_f_measure[attribute]:
                        highest_f_measure[attribute][bias_i_fr] = (
                            discr_i_interval,
                            f_measure,
                        )
                    else:
                        if f_measure > highest_f_measure[attribute][bias_i_fr][1]:
                            # Update best matching interval and highest_f_measure
                            highest_f_measure[attribute][bias_i_fr] = (
                                discr_i_interval,
                                f_measure,
                            )
        # If the attribute is not discretized, by defaul the overalap is 0

    return highest_f_measure, f_measure_dict


def check_tree_interval_similarity_with_bias(
    tree_discr,
    injected_biases,
    apply_generalization=False,
    modality="consider_attribute_hierarchy",
    ver2=False,
):
    # Interval_similarity: Max f-measure injected bias matching,  max matching
    # interval_similarity_detail: Summary stats of injected bias matching --> all matching
    # Overlap_n: summary stats of injected bias matching --> All results of matching
    interval_similarity, interval_similarity_detail, overlap_n = {}, {}, {}
    avg_similarity = {}

    from utils_check_interval_similarity import check_interval_similarity_with_bias

    generalization_dict = tree_discr.generalization_dict
    # Manage also the case of multiple patterns
    for e, injected_bias in enumerate(injected_biases):

        # We initialize the similarity score
        interval_similarity_detail[e], overlap_n[e] = {}, {}

        interval_similarity[e] = {attribute: 0 for attribute in injected_bias}

        # If single tree for each attribute
        if type(tree_discr.trees) is dict:
            # For each discretized attribute
            for attribute, t in tree_discr.trees.items():
                # We check if the attribute is also an attribute for which we inject a bias
                if attribute in injected_bias:

                    # We firstly get all the discretization intervals for that attribute

                    tree_discretization_intervals = t.get_discretization_intervals(
                        generalization_dict=generalization_dict,
                        apply_generalization=apply_generalization,
                        modality=modality,
                        ver2=ver2,
                    )

                    min_max_values = t.min_max_for_attribute

                    # We compute the similarity score based on the f-measure

                    (
                        interval_similarity_i,
                        overlap_n_i,
                    ) = check_interval_similarity_with_bias(
                        injected_bias, tree_discretization_intervals, min_max_values
                    )

                    # We update the similarity interval
                    (
                        interval_similarity_detail[e][attribute],
                        overlap_n[e][attribute],
                    ) = (
                        interval_similarity_i[attribute],
                        overlap_n_i[attribute],
                    )

        else:
            t = tree_discr.trees
            tree_discretization_intervals = t.get_discretization_intervals(
                apply_generalization=apply_generalization,
                generalization_dict=tree_discr.generalization_dict,
                ver2=ver2,
            )
            min_max_values = t.min_max_for_attribute
            (
                interval_similarity_detail[e],
                overlap_n[e],
            ) = check_interval_similarity_with_bias(
                injected_bias, tree_discretization_intervals, min_max_values
            )
        manage_multiple = False
        for attribute, res in interval_similarity_detail[e].items():

            if len(res.values()) == 1:
                interval_similarity[e][attribute] = list(res.values())[0][1]

            elif len(res.values()) > 1:
                manage_multiple = True
                interval_similarity[e][attribute] = [
                    similarity_value for (interval, similarity_value) in res.values()
                ]

        if manage_multiple:
            raise ValueError("TODO manage multiple item intervals injected")
        from statistics import mean

        avg_similarity[e] = mean(interval_similarity[e].values())

    return {
        "highest_f_measure": interval_similarity,
        "highest_f_measure_detail": interval_similarity_detail,
        "all_f_measure_overlaps": overlap_n,
        "average_highest_f_measure": avg_similarity,
    }


def convert_edges_in_discretization_intervals(edges):
    discretization_intervals = {}
    for attribute, values in edges.items():
        discretization_intervals[attribute] = []
        for i in range(0, len(values) - 1):
            discretization_intervals[attribute].append([values[i], values[i + 1]])
    return discretization_intervals


def check_discretization_interval_similarity_with_bias(
    pattern_bias, discretization_intervals, min_max_values
):

    # discretization_intervals = convert_edges_in_discretization_intervals(edges)

    from utils_check_interval_similarity import check_interval_similarity_with_bias

    highest_f_measure_detail, overlap_n = check_interval_similarity_with_bias(
        pattern_bias, discretization_intervals, min_max_values
    )

    highest_f_measure = {attribute: 0 for attribute in pattern_bias}

    manage_multiple = False
    for attribute, res in highest_f_measure_detail.items():

        if len(res.values()) == 1:
            highest_f_measure[attribute] = list(res.values())[0][1]

        elif len(res.values()) > 1:
            manage_multiple = True
            highest_f_measure[attribute] = [
                similarity_value for (interval, similarity_value) in res.values()
            ]
    if manage_multiple:
        raise ValueError("TODO manage multiple item intervals injected")
    from statistics import mean

    avg_similarity = mean(highest_f_measure.values())

    return {
        "highest_f_measure": highest_f_measure,
        "highest_f_measure_detail": highest_f_measure_detail,
        "all_f_measure_overlaps": overlap_n,
        "average_highest_f_measure": avg_similarity,
    }


def get_interval_itemset(itemset, min_max_values):
    intervals_pattern = {}
    for item in itemset:
        v = item.split("=")
        attribute = v[0]
        value = "=".join(v[1:])

        if ("[" in value) and ("]" in value):
            value = value.replace("[", "").replace("]", "")
            v1, v2 = value.split("-")
            interval = [float(v1), float(v2)]
        elif ">=" in value:
            v1 = value.split(">=")[1]
            interval = [float(v1), min_max_values[attribute][1]]
        elif "<=" in value:
            v2 = value.split("<=")[1]
            interval = [min_max_values[attribute][0], float(v2)]
        else:
            print(item)
        intervals_pattern[attribute] = interval

    return intervals_pattern


def get_divergent_itemset_similarity(itemset, pattern_injected_bias, min_max_values):
    intervals_pattern = get_interval_itemset(itemset, min_max_values)
    intervals_pattern = {
        attribute: [interval] for attribute, interval in intervals_pattern.items()
    }

    similarity_info = check_discretization_interval_similarity_with_bias(
        pattern_injected_bias, intervals_pattern, min_max_values
    )
    return similarity_info
