from utils_metric_tree_ranking import entropy


class Splitting_node_ranking:
    from TreeDivergence import Node

    def __init__(
        self,
        data_val_1,
        data_val_2,
        metric_name,
        baseline_metric,
        type_criterion,
        p_k_root=None,
        p_root=None,
        dataset_size=None,
        divergence=None,
        parent_criterion=None,  # TODO CHECK == node criterion
    ):
        self.type_criterion = type_criterion
        self.split_node_1 = Split_node(metric_name)
        self.split_node_2 = Split_node(metric_name)
        parent_node_size = len(data_val_1) + len(data_val_2)
        self.divergence = divergence

        self.metric_value_s = divergence + (p_k_root / p_root)

        self.support = parent_node_size / dataset_size

        self.parent_criterion = parent_criterion

        self.split_node_1.evaluate_node_criterion(
            data_val_1,
            type_criterion,
            baseline_metric,
            p_k_root=p_k_root,
            p_root=p_root,
            dataset_size=dataset_size,
            parent_size=parent_node_size,
            f_over_parent_node=self.metric_value_s,
        )
        self.split_node_2.evaluate_node_criterion(
            data_val_2,
            type_criterion,
            baseline_metric,
            p_k_root=p_k_root,
            p_root=p_root,
            dataset_size=dataset_size,
            parent_size=parent_node_size,
            f_over_parent_node=self.metric_value_s,
        )

        self.split_criterion = None

        self.size_splitting_node = (
            self.split_node_1.node_size + self.split_node_2.node_size
        )

    def evaluate_split_criterion(self, verbose=False):

        if self.type_criterion == "entropy":
            self.split_criterion = (
                self.split_node_1.node_size / self.size_splitting_node
            ) * self.split_node_1.criterion_value + (
                self.split_node_2.node_size / self.size_splitting_node
            ) * self.split_node_2.criterion_value
        elif self.type_criterion == "divergence_criterion":
            self.split_criterion = (
                self.split_node_1.node_size / self.size_splitting_node
            ) * self.split_node_1.criterion_value + (
                self.split_node_2.node_size / self.size_splitting_node
            ) * self.split_node_2.criterion_value

        elif self.type_criterion in ["weighted_sum_abs", "weighted_sum_pow"]:
            self.split_criterion = self.split_node_1.node_support * (
                self.split_node_1.criterion_value
            ) + self.split_node_2.node_support * (self.split_node_2.criterion_value)
            if verbose:
                print(
                    "size_node_1 =",
                    self.split_node_1.node_support,
                    "criterion_node_1 =",
                    self.split_node_1.criterion_value,
                    "size_node_2 =",
                    self.split_node_2.node_support,
                    "criterion_node_2 =",
                    self.split_node_2.criterion_value,
                )

        elif self.type_criterion in ["weighted_max_abs", "weighted_max_pow"]:
            v1 = self.split_node_1.node_support * (self.split_node_1.criterion_value)
            v2 = self.split_node_2.node_support * (self.split_node_2.criterion_value)
            self.split_criterion = max(v1, v2)

        # elif self.type_criterion == "KL_derived":
        #     self.split_criterion = (
        #         self.split_node_1.criterion_value + self.split_node_2.criterion_value
        #     )
        # elif self.type_criterion in ["sum_abs", "sum_pow"]:
        #     self.split_criterion = (
        #         self.split_node_1.criterion_value + self.split_node_2.criterion_value
        #     )
        # elif self.type_criterion in ["log_ratio", "log_ratio_plus_2"]:
        #     self.split_criterion = self.split_node_1.node_support * (
        #         self.split_node_1.criterion_value
        #     ) + self.split_node_2.node_support * (self.split_node_2.criterion_value)

        # elif self.type_criterion in ["weighted_node_sum_abs", "weighted_node_sum_pow"]:
        #     self.split_criterion = (
        #         self.split_node_1.node_size / self.size_splitting_node
        #     ) * (self.split_node_1.criterion_value) + (
        #         self.split_node_2.node_size / self.size_splitting_node
        #     ) * (
        #         self.split_node_2.criterion_value
        #     )
        # elif self.type_criterion == "divergence_pow":
        #     self.split_criterion = (
        #         self.split_node_1.node_size / self.size_splitting_node
        #     ) * self.split_node_1.criterion_value + (
        #         self.split_node_2.node_size / self.size_splitting_node
        #     ) * self.split_node_2.criterion_value
        # elif self.type_criterion in ["max_abs", "max_pow"]:
        #     v1 = self.split_node_1.criterion_value
        #     v2 = self.split_node_2.criterion_value
        #     self.split_criterion = max(v1, v2)

        # elif self.type_criterion in ["weighted_node_max_abs", "weighted_node_max_pow"]:
        #     v1 = (self.split_node_1.node_size / self.size_splitting_node) * (
        #         self.split_node_1.criterion_value
        #     )
        #     v2 = (self.split_node_2.node_size / self.size_splitting_node) * (
        #         self.split_node_2.criterion_value
        #     )
        #     self.split_criterion = max(v1, v2)

        else:
            raise ValueError("{self.type_criterion} measure is not available")
        # print(self.split_node_1.criterion_value, self.split_node_2.criterion_value, self.split_criterion)
        return

    def check_if_best_criterion(
        self, best_criterion, parent_split_criterion_value=None
    ):

        criterion = self.split_criterion

        if (
            parent_split_criterion_value is not None
            and criterion == parent_split_criterion_value
        ):
            return False
        if best_criterion is None:
            return True
        if self.type_criterion == "entropy":
            return criterion < best_criterion
        elif self.type_criterion == "divergence_pow":
            return criterion < best_criterion
        else:  # type_criterion=="KL" or
            return criterion > best_criterion

    def check_gain_divergence(self, parent_divergence):
        divergence_1 = self.split_node_1.divergence
        divergence_2 = self.split_node_2.divergence
        if parent_divergence is None:
            # if not set, TODO remove this check
            return True
        elif (parent_divergence == divergence_1) and (
            parent_divergence == divergence_2
        ):
            return False
        else:
            return True

    def check_gain_criterion(
        self, minimal_gain, parent_criterion, type_weight="support_based", verbose=False
    ):
        criterion_value_1 = self.split_node_1.criterion_value
        criterion_value_2 = self.split_node_2.criterion_value
        # TODO
        #
        parent_criterion = self.parent_criterion
        if self.type_criterion == "entropy":
            gain = self.support * (
                parent_criterion
                - (self.split_node_1.node_support_split * criterion_value_1)
                - (self.split_node_2.node_support_split * criterion_value_2)
            )

            if verbose:
                print(f"GAIN: {gain}")
                print(
                    "support",
                    self.support,
                    "parent_criterion",
                    parent_criterion,
                    "node_support_split 1",
                    self.split_node_1.node_support_split,
                    "criterion_value_1",
                    criterion_value_1,
                    "node_support_split 2",
                    self.split_node_2.node_support_split,
                    "criterion_value_2",
                    criterion_value_2,
                    # "nr",
                    # nr,
                )

            if gain > minimal_gain:
                return True, gain
            else:
                return False, gain
        elif self.type_criterion == "divergence_criterion":

            gain = self.support * self.split_criterion
            if gain > minimal_gain:
                return True, gain
            else:
                return False, gain
        # elif self.type_criterion in ["log_ratio", "log_ratio_plus_2"]:
        #     gain = self.support * (
        #         max(
        #             (self.split_node_1.metric_value),
        #             (self.split_node_2.metric_value),
        #         )
        #         - (self.metric_value_s)
        #     )

        #     verbose = True
        #     if verbose:
        #         print(
        #             "E",
        #             "GAIN =",
        #             gain,
        #             "support =",
        #             self.support,
        #             "node_support_split_1  =",
        #             self.split_node_1.node_support_split,
        #             "divergence_1 =",
        #             (self.split_node_1.divergence),
        #             "criterion_value_1 =",
        #             criterion_value_1,
        #             "node_support_split_2  =",
        #             self.split_node_2.node_support_split,
        #             "divergence_2  =",
        #             (self.split_node_2.divergence),
        #             "criterion_value_2 =",
        #             criterion_value_2,
        #             "divergence = ",
        #             (self.divergence),
        #             "metric_value_1 = ",
        #             self.split_node_1.metric_value,
        #             "metric_value_2 = ",
        #             self.split_node_2.metric_value,
        #             "metric_d = ",
        #             self.metric_value_s,
        #         )

        #     if gain > minimal_gain:
        #         return True, gain
        #     else:
        #         return False, gain

        else:

            # ---------------------- used

            gain = self.support * (
                max(
                    abs(self.split_node_1.divergence),
                    abs(self.split_node_2.divergence),
                )
                - abs(self.divergence)
            )
            verbose = False
            if verbose:
                print(
                    "E",
                    "GAIN =",
                    gain,
                    "support =",
                    self.support,
                    "node_support_split_1  =",
                    self.split_node_1.node_support_split,
                    "divergence_1 =",
                    (self.split_node_1.divergence),
                    "node_support_split_2  =",
                    self.split_node_2.node_support_split,
                    "divergence_2  =",
                    (self.split_node_2.divergence),
                    "divergence",
                    (self.divergence),
                )

            # gain = round(gain, 10)

            if gain > minimal_gain:
                return True, gain
            else:
                return False, gain

        return True, None

    def __str__(self) -> str:
        return f"Node_1:   {self.split_node_1.__str__()} Node_2: {self.split_node_2.__str__()}"


class Split_node:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.divergence = None
        self.criterion_value = None
        self.node_size = None
        self.node_support = None

        self.metric_value = None
        self.divergence = None

    def evaluate_node_criterion(
        self,
        data_val,
        type_criterion,
        baseline_metric,
        p_k_root=None,
        p_root=None,
        dataset_size=None,
        parent_size=None,
        f_over_parent_node=None,
    ):
        from utils_metric_tree_ranking import get_n_i_over_n

        p_i, p_tot = get_n_i_over_n(data_val, metric=self.metric_name)

        p = p_i / p_tot

        self.metric_value = p
        self.divergence = p - baseline_metric
        self.node_size = p_tot

        self.node_support_split = len(data_val) / parent_size

        self.node_support = len(data_val) / dataset_size

        self.criterion_value = compute_criterion(
            type_criterion,
            self.divergence,
            p_i,
            p_tot,
            p_k_root=p_k_root,
            p_root=p_root,
            f_over_parent_node=f_over_parent_node,
        )

    def __str__(self) -> str:
        return f"sup={self.node_support} criterion={self.criterion_value} divergence={self.divergence}"


def compute_criterion(
    type_criterion, divergence, p_i, p_tot, p_k_root, p_root, f_over_parent_node=None
):
    p = p_i / p_tot

    if type_criterion == "entropy":
        from utils_metric_tree_ranking import entropy

        criterion_value = entropy(p) + entropy(1 - p)

    # elif type_criterion == "log_ratio":

    #     p_plus = (p_i + 1) / (p_tot + 1)
    #     q_plus = (p_k_root + 1) / (p_root + 1)

    #     from utils_metric_tree import evaluate_log_ratio

    #     criterion_value = abs(evaluate_log_ratio(p_plus, q_plus))

    # elif type_criterion == "log_ratio_plus_2":

    #     p_plus = (p_i + 1) / (p_tot + 1)
    #     q_plus = (p_k_root + 1) / (p_root + 1)

    #     from utils_metric_tree import evaluate_log_ratio_cap

    #     criterion_value = evaluate_log_ratio_cap(p_plus, q_plus)

    # elif type_criterion == "KL_derived":
    #     from utils_metric_tree import evaluate_KL_derived

    #     w = (p_tot + 1) / (p_root + 1)
    #     p_plus = (p_i + 1) / (p_tot + 1)
    #     q_plus = (p_k_root + 1) / (p_root + 1)
    #     criterion_value = evaluate_KL_derived(w, p_plus, q_plus)

    # elif type_criterion in ["sum_abs", "max_abs"]:
    #     criterion_value = abs(divergence)

    elif type_criterion in [
        "weighted_sum_abs",
        "weighted_max_abs",
        "weighted_node_sum_abs",
    ]:
        criterion_value = abs(divergence)

    elif type_criterion == "divergence_criterion":
        # criterion_value = abs(p - f_over_parent_node)
        criterion_value = abs(p - f_over_parent_node)

    elif type_criterion in ["sum_pow", "max_pow"]:
        criterion_value = (divergence) ** 2
    elif type_criterion in [
        "weighted_sum_pow",
        "weighted_max_pow",
        "weighted_node_sum_pow",
        "weighted_node_max_pow",
    ]:
        criterion_value = (divergence) ** 2
    elif type_criterion == "divergence_pow":
        criterion_value = divergence
    else:
        raise ValueError(f"{type_criterion} is not available")
    return criterion_value
