# from utils_metric_tree import tpr_df
from copy import deepcopy
import operator
from pickle import FALSE


from numpy.core.numeric import False_

PRINT_VERBOSE = False

ops = {
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def get_edges_intervals(edges):
    edges = list(set(edges))
    edges.sort()

    return [
        (edges[i], edges[i + 1])
        for i in range(0, len(edges) - 1, 2)
        if edges[i] != edges[i + 1]
    ]


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


def get_overlap_intervals_point(interval_1, interval_2):

    # |-----------------|
    #        |-----------------|
    #        |----------|         Overlap
    left_edge_overlap = max(interval_1[0], interval_2[0])
    right_edge_overlap = min(interval_1[1], interval_2[1])
    if interval_1[0] == interval_1[1]:
        if (interval_1[0] <= interval_2[1]) and (interval_1[0] >= interval_2[0]):
            return [interval_2[0], interval_1[0]], [interval_1[1], interval_2[1]]

    if interval_2[0] == interval_2[1]:
        if (interval_2[0] <= interval_1[1]) and (interval_2[0] >= interval_1[0]):
            return [interval_1[0], interval_2[0]], [interval_2[1], interval_1[1]]
    return None


def get_overlap_as_interval(interval_1, interval_2):

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
    return [left_edge_overlap, right_edge_overlap]


def _add_info(
    generalization_dict,
    discretizations,
    attribute,
    itemset_name,
    parent_node_name,
    vals,
    rels,
    parent_nodes,
):
    if parent_node_name:
        if attribute not in generalization_dict:
            generalization_dict[attribute] = {}
        generalization_dict[attribute][itemset_name] = parent_node_name

    if itemset_name not in parent_nodes:
        if attribute not in discretizations:
            discretizations[attribute] = {}
        discretizations[attribute][itemset_name] = {
            "rels": rels,
            "vals": vals,
        }
    return (
        generalization_dict,
        discretizations,
    )


def _get_intervals_from_relation(relation, interval):

    for e in range(0, len(relation["rels"])):
        if relation["rels"][e] == ">=":
            interval[0] = relation["vals"][e]
        elif relation["rels"][e] == "<=":
            interval[1] = relation["vals"][e]
        elif relation["rels"][e] == "=":
            interval = [relation["vals"][e], relation["vals"][e]]
        else:
            print(relation, interval)
            print(relation["rels"][e])
            raise ValueError("TODO")
    return interval


def _convert_to_rels_and_name(interval, min_max_attribute):
    rels, vals = [], []

    if interval[0] != min_max_attribute[0]:
        rels.append(">=")
        vals.append(interval[0])
    if interval[1] != min_max_attribute[1]:
        rels.append("<=")
        vals.append(interval[1])
    if len(rels) == 2:
        itemset_name = f"[{interval[0]}-{interval[1]}]"
    else:
        itemset_name = f"{rels[0]}{vals[0]}"
    return itemset_name, rels, vals


class TreeDivergence:
    def __init__(self):
        self.tree = None
        self.metric = None
        self.dataset_len = None
        self.minimum_support = None
        self.root_metric = None
        self.attributes = None
        self.criterion = "sum_abs"  # default
        self.minimal_gain = None
        self.min_max_for_attribute = {}
        self.number_internal_nodes = None
        self.number_leaf_nodes = None

    def generateTree(
        self,
        data,
        class_map,
        class_name="class",
        pred_name="predicted",
        metric="d_fpr",
        minimum_support=0.1,
        verbose=False,
        **kwargs,
    ):

        from utils_metric_tree import instanceConfusionMatrix, getPerformanceMetric

        attributes = [a for a in list(data.columns) if a not in [class_name, pred_name]]

        for attribute in attributes:
            self.min_max_for_attribute[attribute] = [
                min(data[attribute]),
                max(data[attribute]),
            ]

        data = (
            instanceConfusionMatrix(
                data, class_map, class_name=class_name, pred_name=pred_name
            )
            .drop(columns=[class_name, pred_name])
            .copy()
        )
        self.dataset_len = len(data)
        self.minimum_support = minimum_support
        self.metric = metric
        self.setDiscreteContinuosAttributes(data, attributes)
        self.root_metric = getPerformanceMetric(data, metric=self.metric)

        from utils_metric_tree import get_n_i_over_n

        data_values = data.drop(columns=["tn", "fp", "fn", "tp"]).copy()
        conf_matrix_rows = data[["tn", "fp", "fn", "tp"]]
        self.p_k_root, self.p_root = get_n_i_over_n(
            conf_matrix_rows.values, metric=self.metric
        )

        self.attributes = list(data_values.columns)
        from utils_metric_tree import getDivergenceMetric_np

        node_divergence = getDivergenceMetric_np(
            conf_matrix_rows.values,
            metric=self.metric,
            metric_baseline=self.root_metric,
        )
        from utils_metric_tree import getPerformanceMetric_np

        node_metric = getPerformanceMetric_np(
            conf_matrix_rows.values,
            metric=self.metric,
            cols_orderTP_dict={"tn": 0, "fp": 1, "fn": 2, "tp": 3},
        )

        # TODO
        if "type_criterion" in kwargs:
            self.criterion = kwargs["type_criterion"]

        if "minimal_gain" in kwargs:
            self.minimal_gain = kwargs["minimal_gain"]

        from splitting_node import compute_criterion

        node_criterion = compute_criterion(
            self.criterion,
            node_divergence,
            self.p_k_root,
            self.p_root,
            self.p_k_root,
            self.p_root,
            f_over_parent_node=node_metric,
        )

        self.tree = self.buildTree(
            data_values.values,
            conf_matrix_rows.values,
            "root",
            "",
            "",
            node_divergence,
            verbose=verbose,
            node_criterion=node_criterion,
            parent_size=self.dataset_len,
            **kwargs,
        )

        return self.tree

    def buildTree(
        self,
        data_values,
        data_conf_matrix_values,
        attr,
        val,
        rel,
        node_divergence,
        verbose=False,
        node_criterion=None,
        type_criterion="sum_abs",
        gain=False,
        parent_split_criterion_value=None,
        minimal_gain=None,
        parent_size=None,
        node_split_value=None,
    ):

        from utils_metric_tree import getConfusionMatrix_np

        node_size = len(data_values)
        node_data = Node(
            node_divergence,
            node_size / self.dataset_len,
            attr,
            val,
            rel,
            confusion_matrix=getConfusionMatrix_np(data_conf_matrix_values),
            node_criterion=node_criterion,
            node_support_ratio=node_size / parent_size,
            node_split_value=node_split_value,
        )
        if type_criterion == "entropy":
            from utils_metric_tree import entropy_node

            node_data.set_entropy(
                entropy_node(data_conf_matrix_values, metric=self.metric)
            )
        from utils_metric_tree import get_measure_node

        node_data.set_measure_node(
            get_measure_node(data_conf_matrix_values, metric=self.metric)
        )

        best_split_info = self.selectBestSplit(
            data_values,
            data_conf_matrix_values,
            self.dataset_len,
            min_support=self.minimum_support,
            verbose=verbose,
            type_criterion=type_criterion,
            gain=gain,
            parent_split_criterion_value=parent_split_criterion_value,
            parent_divergence=node_divergence,
            minimal_gain=self.minimal_gain,
            parent_criterion=node_criterion,
        )

        if best_split_info is not None:
            node_data.set_criterion_split(best_split_info["criterion"])
            # TODO DELETE
            if (minimal_gain is not None) and ("gain" in best_split_info):
                node_data.set_gain(best_split_info["gain"])

        if best_split_info is not None:
            if best_split_info["rel"][0] == "<=":
                rel_val_2 = ">"
            elif best_split_info["rel"][0] == "=":
                rel_val_2 = "!="
            elif best_split_info["rel"][0] == "<":
                rel_val_2 = ">="
            else:
                raise ValueError(best_split_info["rel"][0])
            node_data.children = [
                self.buildTree(
                    data_values[best_split_info["indexes"][0]],
                    data_conf_matrix_values[best_split_info["indexes"][0]],
                    best_split_info["attr"],
                    best_split_info["vals"][0],
                    best_split_info["rel"][0],
                    best_split_info["divergence"][0],
                    verbose=verbose,
                    type_criterion=type_criterion,
                    gain=gain,
                    parent_split_criterion_value=node_data.criterion_split,
                    node_criterion=best_split_info["criterion_nodes"][0],
                    minimal_gain=minimal_gain,
                    parent_size=node_size,
                    node_split_value=[
                        best_split_info["attr"],
                        best_split_info["rel"][0],
                        best_split_info["vals"][0],
                    ],
                ),
                self.buildTree(
                    data_values[best_split_info["indexes"][1]],
                    data_conf_matrix_values[best_split_info["indexes"][1]],
                    best_split_info["attr"],
                    best_split_info["vals"][1],
                    best_split_info["rel"][1],
                    best_split_info["divergence"][1],
                    verbose=verbose,
                    type_criterion=type_criterion,
                    gain=gain,
                    parent_split_criterion_value=node_data.criterion_split,
                    node_criterion=best_split_info["criterion_nodes"][1],
                    minimal_gain=minimal_gain,
                    parent_size=node_size,
                    node_split_value=[
                        best_split_info["attr"],
                        rel_val_2,
                        best_split_info["vals"][0],
                    ],
                ),
            ]
        else:
            if verbose:
                print("********************** NONE")
                print()
                print(node_data)
        return node_data

    def selectBestSplit(
        self,
        data_values,
        data_conf_matrix_values,
        dataset_len,
        min_support=0.1,
        type_criterion="sum_abs",
        gain=False,
        gain_parent=True,
        verbose=False,
        parent_split_criterion_value=None,
        parent_divergence=None,
        minimal_gain=None,
        parent_criterion=None,
    ):
        import numpy as np
        from utils_metric_tree import (
            isAttributeBinary,
            getSupport,
            getReciprocal,
        )

        best_split_info = None
        best_criterion = None

        gain_value = None

        if getSupport(data_values, dataset_len) < min_support:
            return best_split_info
        for attr_id, attr_name in enumerate(self.attributes):
            if getSupport(data_values[:, attr_id], dataset_len) < min_support:
                continue

            attr_array = data_values[:, attr_id]
            if attr_name in self.discrete_attr:
                attr_vals = np.unique(attr_array)
                for val in attr_vals:
                    # Discrete splitting
                    id_eq = np.where(attr_array == val)
                    id_diff = np.where(attr_array != val)
                    part_1 = data_values[id_eq]
                    part_2 = data_values[id_diff]
                    sup_1 = getSupport(part_1, dataset_len)
                    sup_2 = getSupport(part_2, dataset_len)

                    if (sup_1 < min_support) or (sup_2 < min_support):
                        continue

                    data_val_1 = data_conf_matrix_values[id_eq]
                    data_val_2 = data_conf_matrix_values[id_diff]

                    from splitting_node import Splitting_node

                    split_node = Splitting_node(
                        data_val_1,
                        data_val_2,
                        self.metric,
                        self.root_metric,
                        type_criterion=type_criterion,
                        p_k_root=self.p_k_root,
                        p_root=self.p_root,
                        dataset_size=dataset_len,
                        divergence=parent_divergence,
                        parent_criterion=parent_criterion,
                    )

                    split_node.evaluate_split_criterion()

                    is_best_split = split_node.check_if_best_criterion(
                        best_criterion, parent_split_criterion_value
                    )
                    PRINT_VERBOSE = False
                    if PRINT_VERBOSE:
                        print(
                            "E",
                            "attr = ",
                            attr_name,
                            "vals = ",
                            (val_1, val_2),
                            "rel = ",
                            ("<=", ">="),
                            "criterion = ",
                            split_node.split_criterion,
                            "values = ",
                            [
                                split_node.split_node_1.metric_value,
                                split_node.split_node_2.metric_value,
                            ],
                            "divergence = ",
                            [
                                split_node.split_node_1.divergence,
                                split_node.split_node_2.divergence,
                            ],
                            "support_nodes = ",
                            [
                                split_node.split_node_1.node_support_split,
                                split_node.split_node_2.node_support_split,
                            ],
                            "criterion_nodes = ",
                            [
                                split_node.split_node_1.criterion_value,
                                split_node.split_node_2.criterion_value,
                            ],
                            "is_best_split =",
                            is_best_split,
                            "\n",
                        )

                    if is_best_split:
                        best_criterion = split_node.split_criterion
                        isbinary = isAttributeBinary(attr_vals)
                        val2 = getReciprocal(attr_vals, val) if isbinary else val
                        rel2 = "=" if isbinary else "!="
                        best_split_info = {
                            "attr": attr_name,
                            "vals": (val, val2),
                            "rel": ("=", rel2),
                            "indexes": [
                                id_eq,
                                id_diff,
                            ],
                            "criterion": split_node.split_criterion,
                            "divergence": [
                                split_node.split_node_1.divergence,
                                split_node.split_node_2.divergence,
                            ],
                            "criterion_nodes": [
                                split_node.split_node_1.criterion_value,
                                split_node.split_node_2.criterion_value,
                            ],
                            "split_node": split_node,
                        }
                        if gain_value is not None:
                            best_split_info["gain"] = gain_value

                    if isAttributeBinary(attr_vals):
                        break
            else:

                from copy import deepcopy

                index_sorted = np.argsort(data_values[:, attr_id])
                data_attr_sorted = data_values[index_sorted, attr_id]

                start_i = int(len(data_values) * min_support) - 1

                for i in range(start_i, len(data_values) - 1):
                    if data_attr_sorted[i] != data_attr_sorted[i + 1]:
                        val_1 = data_attr_sorted[i]
                        val_2 = data_attr_sorted[i + 1]

                        sup_1 = getSupport(data_attr_sorted[: i + 1], dataset_len)
                        sup_2 = getSupport(data_attr_sorted[i + 1 :], dataset_len)
                        if sup_1 < min_support:
                            continue
                        if sup_2 < min_support:
                            break

                        data_val_1 = data_conf_matrix_values[index_sorted[: i + 1]]
                        data_val_2 = data_conf_matrix_values[index_sorted[i + 1 :]]

                        from splitting_node import Splitting_node

                        split_node = Splitting_node(
                            data_val_1,
                            data_val_2,
                            self.metric,
                            self.root_metric,
                            type_criterion=type_criterion,
                            p_k_root=self.p_k_root,
                            p_root=self.p_root,
                            dataset_size=dataset_len,
                            divergence=parent_divergence,
                            parent_criterion=parent_criterion,
                        )

                        split_node.evaluate_split_criterion()

                        is_best_split = split_node.check_if_best_criterion(
                            best_criterion, parent_split_criterion_value
                        )
                        PRINT_VERBOSE = False
                        if PRINT_VERBOSE:
                            print(
                                "E",
                                "attr = ",
                                attr_name,
                                "vals = ",
                                (val_1, val_2),
                                "rel = ",
                                ("<=", ">="),
                                "criterion = ",
                                split_node.split_criterion,
                                "values = ",
                                [
                                    split_node.split_node_1.metric_value,
                                    split_node.split_node_2.metric_value,
                                ],
                                "support_nodes = ",
                                [
                                    split_node.split_node_1.node_support_split,
                                    split_node.split_node_2.node_support_split,
                                ],
                                "divergence = ",
                                [
                                    split_node.split_node_1.divergence,
                                    split_node.split_node_2.divergence,
                                ],
                                "criterion_nodes = ",
                                [
                                    split_node.split_node_1.criterion_value,
                                    split_node.split_node_2.criterion_value,
                                ],
                                "is_best_split =",
                                is_best_split,
                                "\n",
                            )

                        if is_best_split:

                            best_criterion = split_node.split_criterion

                            best_split_info = {
                                "attr": attr_name,
                                "vals": (val_1, val_2),
                                "rel": ("<=", ">="),
                                "indexes": [
                                    index_sorted[: i + 1],
                                    index_sorted[i + 1 :],
                                ],
                                "criterion": split_node.split_criterion,
                                "divergence": [
                                    split_node.split_node_1.divergence,
                                    split_node.split_node_2.divergence,
                                ],
                                "criterion_nodes": [
                                    split_node.split_node_1.criterion_value,
                                    split_node.split_node_2.criterion_value,
                                ],
                                "split_node": split_node,
                            }
                            if gain_value is not None:
                                best_split_info["gain"] = gain_value

        if best_split_info is not None:
            is_best_split = best_split_info["split_node"].check_gain_divergence(
                best_split_info["split_node"].divergence
            )
            if (
                is_best_split
                and (minimal_gain is not None)
                # and (type_criterion == "entropy")
            ):

                is_best_split, gain_value = best_split_info[
                    "split_node"
                ].check_gain_criterion(
                    minimal_gain, best_split_info["split_node"].parent_criterion
                )

                if is_best_split:
                    best_split_info["gain"] = gain_value

            if is_best_split == False:
                best_split_info = None

        PRINT_VERBOSE = False

        if PRINT_VERBOSE:
            if best_split_info is not None:
                is_best_split, gain = best_split_info[
                    "split_node"
                ].check_gain_criterion(
                    minimal_gain, parent_criterion, "support_based", verbose=True
                )
                print("split_criterion:", best_split_info["split_node"].split_criterion)
                print("Gain: ", gain)
                print("parent_criterion", parent_criterion)
                best_split_info["split_node"].evaluate_split_criterion(verbose=True)
                pr = [
                    "attr",
                    "vals",
                    "rel",
                    "criterion",
                    "divergence",
                    "criterion_nodes",
                ]
                print({p: best_split_info[p] for p in pr})
                print(best_split_info["split_node"])
                print()
        return best_split_info

    def printTree(self, tree=None, round_v=5, show_condition=False):
        tree = tree if tree else self.tree
        self.printNode(tree, round_v=round_v, show_condition=show_condition)

    def printNode(self, node, indent="", round_v=5, show_condition=False):
        if show_condition:
            attr, rel, val = node.node_condition()

            print(
                indent,
                f"{attr}{rel}{val} s={node.support:.{round_v}f} --> {self.metric}={node.metric:.{round_v}f}",
            )
            # print(indent, node.confusion_matrix)
            # print(indent, "Measure node", node.measure_node)
            # print(indent, "Criterion value", node.criterion_value)
            # print(indent)
        else:
            print(
                indent,
                f"{node.attr}{node.rel}{node.val} s={node.support:.{round_v}f} --> {self.metric}={node.metric:.{round_v}f}",
            )

        if node.children:
            for child in node.children:
                self.printNode(
                    child,
                    indent + "        ",
                    round_v=round_v,
                    show_condition=show_condition,
                )

    def printTreeDF(self, tree=None):
        tree = tree if tree else self.tree
        tree_df = []
        self.printNodeDF(tree, tree_df)
        import pandas as pd

        return pd.DataFrame.from_dict(tree_df)

    def printNodeDF(self, node, tree_df, parent_node=[], indent=""):

        isRoot = True if node.attr == "root" else False
        current_node = f"{node.attr}{node.rel}{node.val}"
        node_info = {
            "itemset": frozenset(parent_node + [current_node])
            if isRoot is False
            else frozenset(),
            "support": node.support,
            self.metric: node.metric,
        }
        node_info.update(node.confusion_matrix.copy())

        tree_df.append(node_info)
        if node.children:
            for child in node.children:
                self.printNodeDF(
                    child,
                    tree_df,
                    parent_node=parent_node + [current_node] if isRoot is False else [],
                    indent=indent + "        ",
                )

    def printTreeHierarchy(self, tree=None):
        tree = tree if tree else self.tree
        tree_df = []
        self.printNodeHierarchy(tree, tree_df)
        import pandas as pd

        return pd.DataFrame.from_dict(tree_df)

    def printNodeHierarchy(
        self,
        node,
        tree_df,
        parent_node=[],
        generalized_value="",
        parent_node_info={},
        indent="",
        ver2=False,
    ):

        isRoot = True if node.attr == "root" else False
        current_node = f"{node.attr}{node.rel}{node.val}"
        if ver2:
            (
                rels,
                new_itemset,
                rels_ret,
                vals_ret,
                value_short,
            ) = self.summarizeInterval_v2(node, parent_node_info, isRoot)
        else:
            rels, new_itemset, rels_ret, vals_ret, value_short = self.summarizeInterval(
                node, parent_node_info, isRoot
            )
        if isRoot is False:

            node_info = {
                "attribute": node.attr,
                "parent": parent_node,
                "itemset": f"{node.attr}{node.rel}{node.val}",
                "itemset_summarized": frozenset(new_itemset),
                "detailed_value": f"{node.attr}{node.rel}{node.val}",
                "generalized_value": generalized_value,
                "rel": {node.rel},
                "value": {node.val},
                self.metric: node.metric,
                "level": len(parent_node),
                "hasChild": True if node.children else False,
                "rels_ret": rels_ret,
                "vals_ret": vals_ret,
                "new_value": new_itemset[0].split(node.attr)[1],
            }

            tree_df.append(node_info)
        if node.children:
            for child in node.children:
                self.printNodeHierarchy(
                    child,
                    tree_df,
                    parent_node=parent_node + [current_node] if isRoot is False else [],
                    generalized_value=current_node if isRoot is False else [],
                    indent=indent + "        ",
                    parent_node_info=rels,
                )

    def setDiscreteContinuosAttributes(self, data, attributes, n_discr=10):
        import numpy as np

        self.discrete_attr = []
        self.continuous_attr = []
        for attr in data[attributes]:
            if (data.dtypes[attr] == object) and (len(data[attr].unique()) <= n_discr):
                self.discrete_attr.append(attr)
            else:
                self.continuous_attr.append(attr)

    def summarizeInterval_v2(self, node, parent_node_info, isRoot):
        from copy import deepcopy

        verbose = False
        rels = deepcopy(parent_node_info)
        new_itemset = []
        rels_ret, vals_ret, value_short = {}, {}, {}

        if isRoot:
            return rels, new_itemset, rels_ret, vals_ret, value_short

        if node.attr not in rels:
            rels[node.attr] = []
        # else:
        #    toUpdate = True
        node_attr = {"attr": node.attr, "rel": node.rel, "val": node.val}
        rels[node.attr].append(node_attr)

        for attr, list_discr in rels.items():
            if attr not in rels_ret:
                rels_ret[attr] = []
                vals_ret[attr] = []
                value_short[attr] = []
            if verbose:
                print(attr)
            if len(list_discr) == 1:
                new_itemset.append(
                    f'{list_discr[0]["attr"]}{list_discr[0]["rel"]}{list_discr[0]["val"]}'
                )
                value_short[attr].append(
                    f'{list_discr[0]["rel"]}{list_discr[0]["val"]}'
                )
                rels_ret[attr].append(list_discr[0]["rel"])
                vals_ret[attr].append(list_discr[0]["val"])
            else:
                attr_rel = {}
                eq_relations = [
                    f'{h["attr"]}{h["rel"]}{h["val"]}'
                    for h in list_discr
                    if "=" == h["rel"]
                ]
                if eq_relations:
                    new_itemset.append(eq_relations[0])
                    value_short[attr].append(
                        [f'{h["val"]}' for h in list_discr if "=" == h["rel"]][0]
                    )
                    # TODO check
                    for h in list_discr:
                        rels_ret[attr].append(h["rel"])
                        vals_ret[attr].append(h["val"])
                else:
                    diff_relations = [
                        f'{h["attr"]}{h["rel"]}{h["val"]}'
                        for h in list_discr
                        if "!=" == h["rel"]
                    ]
                    if len(diff_relations) == len(list_discr):
                        new_itemset.extend(diff_relations)
                        for h in list_discr:
                            if "!=" == h["rel"]:
                                rels_ret[attr].append(h["rel"])
                                vals_ret[attr].append(h["val"])
                                value_short[attr].append(f'{h["rel"]}{h["val"]}')
                    else:
                        for d in list_discr:
                            if d["rel"] not in ["<=", ">", "="]:
                                print(attr, d["rel"])
                                raise ValueError("# TODO - only intervals")
                            if d["rel"] not in attr_rel:
                                attr_rel[d["rel"]] = []
                            attr_rel[d["rel"]].append(d["val"])

                        if "<=" in attr_rel:
                            r1 = "<="
                            attr_rel["<="] = min(attr_rel["<="])
                        if ">" in attr_rel:
                            r1 = ">"
                            attr_rel[">"] = max(attr_rel[">"])

                        if "<=" in attr_rel and ">" in attr_rel:
                            # if attr_rel[">"] == attr_rel["<="]:
                            # new_itemset.append(f'{attr}={attr_rel["<="]}')
                            # rels_ret[attr].append("=")
                            # vals_ret[attr].append(attr_rel[">"])
                            # value_short[attr].append(f'{attr_rel[">"]}')
                            # else:
                            new_itemset.append(
                                f'{attr}=[{attr_rel[">"]}-{attr_rel["<="]}]'
                            )
                            # f'{attr_rel[">"]}<={attr}<={attr_rel["<="]}'
                            rels_ret[attr].append(">")
                            vals_ret[attr].append(attr_rel[">"])
                            rels_ret[attr].append("<=")
                            vals_ret[attr].append(attr_rel["<="])
                            value_short[attr].append(
                                f'[{attr_rel[">"]}-{attr_rel["<="]}]'
                            )

                        else:
                            new_itemset.append(f"{attr}{r1}{attr_rel[r1]}")
                            rels_ret[attr].append(r1)
                            vals_ret[attr].append(attr_rel[r1])
                            value_short[attr].append(f"{r1}{attr_rel[r1]}")

        return rels, new_itemset, rels_ret, vals_ret, value_short

    # TODO summarize REWRITE TODO TODO TODO
    # TODO: handle discrete attributes
    def summarizeInterval(self, node, parent_node_info, isRoot):
        from copy import deepcopy

        verbose = False
        rels = deepcopy(parent_node_info)
        new_itemset = []
        rels_ret, vals_ret, value_short = {}, {}, {}

        if isRoot:
            return rels, new_itemset, rels_ret, vals_ret, value_short

        if node.attr not in rels:
            rels[node.attr] = []
        # else:
        #    toUpdate = True
        node_attr = {"attr": node.attr, "rel": node.rel, "val": node.val}
        rels[node.attr].append(node_attr)

        for attr, list_discr in rels.items():
            if attr not in rels_ret:
                rels_ret[attr] = []
                vals_ret[attr] = []
                value_short[attr] = []
            if verbose:
                print(attr)
            if len(list_discr) == 1:
                new_itemset.append(
                    f'{list_discr[0]["attr"]}{list_discr[0]["rel"]}{list_discr[0]["val"]}'
                )
                value_short[attr].append(
                    f'{list_discr[0]["rel"]}{list_discr[0]["val"]}'
                )
                rels_ret[attr].append(list_discr[0]["rel"])
                vals_ret[attr].append(list_discr[0]["val"])
            else:
                attr_rel = {}
                eq_relations = [
                    f'{h["attr"]}{h["rel"]}{h["val"]}'
                    for h in list_discr
                    if "=" == h["rel"]
                ]
                if eq_relations:
                    new_itemset.append(eq_relations[0])
                    value_short[attr].append(
                        [f'{h["val"]}' for h in list_discr if "=" == h["rel"]][0]
                    )
                    # TODO check
                    for h in list_discr:
                        rels_ret[attr].append(h["rel"])
                        vals_ret[attr].append(h["val"])
                else:
                    diff_relations = [
                        f'{h["attr"]}{h["rel"]}{h["val"]}'
                        for h in list_discr
                        if "!=" == h["rel"]
                    ]
                    if len(diff_relations) == len(list_discr):
                        new_itemset.extend(diff_relations)
                        for h in list_discr:
                            if "!=" == h["rel"]:
                                rels_ret[attr].append(h["rel"])
                                vals_ret[attr].append(h["val"])
                                value_short[attr].append(f'{h["rel"]}{h["val"]}')
                    else:
                        for d in list_discr:
                            if d["rel"] not in ["<=", ">=", "="]:
                                print(attr, d["rel"])
                                raise ValueError("# TODO - only intervals")
                            if d["rel"] not in attr_rel:
                                attr_rel[d["rel"]] = []
                            attr_rel[d["rel"]].append(d["val"])

                        if "<=" in attr_rel:
                            r1 = "<="
                            attr_rel["<="] = min(attr_rel["<="])
                        if ">=" in attr_rel:
                            r1 = ">="
                            attr_rel[">="] = max(attr_rel[">="])

                        if "<=" in attr_rel and ">=" in attr_rel:
                            if attr_rel[">="] == attr_rel["<="]:
                                new_itemset.append(f'{attr}={attr_rel["<="]}')
                                rels_ret[attr].append("=")
                                vals_ret[attr].append(attr_rel[">="])
                                value_short[attr].append(f'{attr_rel[">="]}')
                            else:
                                new_itemset.append(
                                    f'{attr}=[{attr_rel[">="]}-{attr_rel["<="]}]'
                                )
                                # f'{attr_rel[">="]}<={attr}<={attr_rel["<="]}'
                                rels_ret[attr].append(">=")
                                vals_ret[attr].append(attr_rel[">="])
                                rels_ret[attr].append("<=")
                                vals_ret[attr].append(attr_rel["<="])
                                value_short[attr].append(
                                    f'[{attr_rel[">="]}-{attr_rel["<="]}]'
                                )

                        else:
                            new_itemset.append(f"{attr}{r1}{attr_rel[r1]}")
                            rels_ret[attr].append(r1)
                            vals_ret[attr].append(attr_rel[r1])
                            value_short[attr].append(f"{r1}{attr_rel[r1]}")

        return rels, new_itemset, rels_ret, vals_ret, value_short

    # TODO
    def printTreeDFSummarized(self, tree=None):
        tree = tree if tree else self.tree
        tree_df = []
        self.printNodeDFSummarized(tree, tree_df)
        import pandas as pd

        return pd.DataFrame.from_dict(tree_df)

    # TODO
    def printNodeDFSummarized(
        self, node, tree_df, parent_node=[], parent_node_info={}, indent="", ver2=False
    ):

        isRoot = True if node.attr == "root" else False
        current_node = f"{node.attr}{node.rel}{node.val}"

        # print(parent_node_info)

        if ver2:
            (
                rels,
                new_itemset,
                rels_ret,
                vals_ret,
                value_short,
            ) = self.summarizeInterval_v2(node, parent_node_info, isRoot)
        else:
            rels, new_itemset, rels_ret, vals_ret, value_short = self.summarizeInterval(
                node, parent_node_info, isRoot
            )
        node_info = {
            "itemset": frozenset(parent_node + [current_node])
            if isRoot is False
            else frozenset(),
            "itemset_short": frozenset(new_itemset),
            "support": node.support,
            self.metric: node.metric,
            "rels_ret": rels_ret,
            "vals_ret": vals_ret,
        }
        node_info.update(node.confusion_matrix.copy())

        tree_df.append(node_info)
        if node.children:
            for child in node.children:
                self.printNodeDFSummarized(
                    child,
                    tree_df,
                    parent_node=parent_node + [current_node] if isRoot is False else [],
                    parent_node_info=rels,
                    indent=indent + "        ",
                )

    # TODO
    def get_hierarchy_DF(self, tree=None, verbose_info=False, ver2=False):
        tree = tree if tree else self.tree
        tree_df = []

        # print("Version ", ver2)

        self.node_get_hierarchy_DF(tree, tree_df, parent_node_names=[], ver2=ver2)

        if tree_df == []:
            return None
        import pandas as pd

        df = pd.DataFrame.from_dict(tree_df)
        if verbose_info:
            return df
        else:
            # TODO
            # Remove the one not in the list -->  keep parent_node_name for now
            info_cols = [
                "attribute",
                "itemset_name",
                "parent_node_name",
                "parent_node_name_all_attrs",
                # "hasChild",
                "rels",
                "vals",
                "level",
            ]
            return df[info_cols]

    # TODO
    def get_discretization_relations_OLD(self, tree=None):
        tree = tree if tree else self.tree
        tree_list = []

        self.node_get_hierarchy_DF(self.tree, tree_list, parent_node_names=[])

        discr_attribute = {}
        for discr_i in tree_list:
            if discr_i["attribute"] not in discr_attribute:
                discr_attribute[discr_i["attribute"]] = []
            discr_attribute[discr_i["attribute"]].append(
                (discr_i["rels"], discr_i["vals"])
            )

        return discr_attribute

    def get_discretization_relations(
        self,
        tree=None,
        apply_generalization=False,
        generalization_dict=None,
        modality="consider_attribute_hierarchy",
        ver2=False,
    ):
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

        def check_add_interval(
            discr_i,
            apply_generalization,
            generalization_dict,
            modality="consider_attribute_hierarchy",
            ver2=False,
        ):
            if apply_generalization == True:
                return True
            else:
                if modality != "consider_attribute_hierarchy":
                    return discr_i["hasChild"] == False
                else:
                    if generalization_dict is None:
                        raise ValueError("TODO")
                    attribute = discr_i["attribute"]
                    itemset_name = discr_i["itemset_name"]
                    if attribute not in generalization_dict:
                        return True
                    else:
                        if itemset_name not in generalization_dict[attribute].values():
                            return True
                        else:
                            return False

        tree = tree if tree else self.tree
        tree_list = []

        self.node_get_hierarchy_DF(
            self.tree, tree_list, parent_node_names=[], ver2=ver2
        )
        # t.min_max_for_attribute
        discr_attribute = {}
        for discr_i in tree_list:
            add_interval = check_add_interval(
                discr_i,
                apply_generalization,
                generalization_dict,
                modality=modality,
                ver2=ver2,
            )
            if add_interval:
                attribute = discr_i["attribute"]
                if attribute not in discr_attribute:
                    discr_attribute[attribute] = []
                if len(discr_i["vals"]) == 1:
                    discr_interval_i = convert_to_interval(
                        zip(discr_i["rels"], discr_i["vals"]),
                        self.min_max_for_attribute[attribute][0],
                        self.min_max_for_attribute[attribute][1],
                    )
                else:
                    discr_interval_i = discr_i["vals"]

                discr_attribute[attribute].append(
                    (discr_i["rels"], discr_i["vals"], discr_interval_i)
                )

        return discr_attribute

    def get_discretization_intervals(
        self,
        tree=None,
        apply_generalization=False,
        generalization_dict=None,
        modality="consider_attribute_hierarchy",
        ver2=False,
    ):
        discr_attribute_interval = self.get_discretization_relations(
            tree=tree,
            generalization_dict=generalization_dict,
            apply_generalization=apply_generalization,
            modality=modality,
            ver2=ver2,
        )
        for attribute in discr_attribute_interval:
            discr_attribute_interval[attribute] = [
                interval for rel, val, interval in discr_attribute_interval[attribute]
            ]
        return discr_attribute_interval

    # TODO
    def node_get_hierarchy_DF(
        self,
        node,
        tree_df,
        parent_node=[],
        parent_node_info={},
        parent_node_name=[],
        parent_node_names=[],
        indent="",
        ver2=False,
    ):
        isRoot = True if node.attr == "root" else False
        current_node = f"{node.attr}{node.rel}{node.val}"
        if ver2:
            (
                rels,
                itemset_attribute_value_short,
                rels_ret,
                vals_ret,
                itemset_value_name,
            ) = self.summarizeInterval_v2(node, parent_node_info, isRoot)

        else:
            (
                rels,
                itemset_attribute_value_short,
                rels_ret,
                vals_ret,
                itemset_value_name,
            ) = self.summarizeInterval(node, parent_node_info, isRoot)
        parent_node_name_attribute = []
        index_i = -1
        if isRoot is False:
            itemset_value_name = itemset_value_name[node.attr]
            import copy

            if parent_node_names:
                indexes = [
                    e
                    for (e, (attr, value)) in enumerate(parent_node_names)
                    if node.attr == attr
                ]
                if indexes != []:
                    index_i = indexes[0]
                    parent_node_name_attribute = parent_node_names[index_i][1]
                    if type(parent_node_name_attribute) is list:
                        parent_node_name_attribute = parent_node_name_attribute.copy()
                        if len(parent_node_name_attribute) == 1:
                            parent_node_name_attribute = parent_node_name_attribute[0]
                    else:
                        parent_node_name_attribute = parent_node_name_attribute

        if type(itemset_value_name) is list and len(itemset_value_name) == 1:
            # itemset_name_short = itemset_attribute_value_short[0]
            itemset_value_name = itemset_value_name[0]

        if type(itemset_value_name) is list and len(itemset_value_name) > 1:
            sep = ", "
            itemset_value_name_new = sep.join(itemset_value_name)
            import warnings

            usermsg = (
                f"We merge the terms {itemset_value_name} into {itemset_value_name_new}"
            )
            warnings.warn(usermsg)
            # itemset_name_short = itemset_attribute_value_short[0]
            itemset_value_name = itemset_value_name_new

        if isRoot is False:
            node_info = {
                "attribute": node.attr,
                "parent": parent_node,
                "parent_node_name_all_attrs": parent_node_name,
                "parent_node_name": frozenset()
                if parent_node_name_attribute == []
                else parent_node_name_attribute,
                "itemset": frozenset(parent_node + [current_node])
                if isRoot is False
                else frozenset(),
                "itemset_short": itemset_attribute_value_short,
                "itemset_name": itemset_value_name,
                "hasChild": True if node.children else False,
                "rels": rels_ret[node.attr],
                "vals": vals_ret[node.attr],
                "level": len(parent_node),
            }
            node.set_item_name(itemset_value_name)
            # node_info.update(node.confusion_matrix.copy())

            tree_df.append(node_info)
        if node.children:
            # parent_node_names[node.attr] = itemset_value_name
            # Alternative: still deepcopy but dictionary? #TODO
            parent_node_names = deepcopy(parent_node_names)
            if index_i == -1:
                # The attribute is not present: add it with an append
                parent_node_names.append((node.attr, itemset_value_name))
            else:
                # Substitute it
                parent_node_names[index_i] = (node.attr, itemset_value_name)
            for child in node.children:
                self.node_get_hierarchy_DF(
                    child,
                    tree_df,
                    parent_node=parent_node + [current_node] if isRoot is False else [],
                    parent_node_name=parent_node_names,
                    parent_node_info=rels,
                    parent_node_names=parent_node_names,
                    indent=indent + "        ",
                    ver2=ver2,
                )

    def getHierarchyAndDiscretizationSplits(self, tree, verbose=False, ver2=False):
        # TODO: manage empy tree

        # print("Version ", ver2)
        df_hierarchy = self.get_hierarchy_DF(tree, ver2)
        if df_hierarchy is None:
            return None, None
        generalization_dict = {}
        generalization_dict_2 = {}
        discretizations = {}
        for attribute, df_hierarchy_attribute in df_hierarchy.groupby("attribute"):
            if not df_hierarchy_attribute["itemset_name"].is_unique:
                import warnings

                if verbose:
                    print(
                        df_hierarchy_attribute[
                            df_hierarchy_attribute.duplicated(
                                ["itemset_name"], keep=False
                            )
                        ]
                    )

                warnings.warn(
                    f"Attribute {attribute} splitted in the same way multiple ways."
                )
            parent_nodes = df_hierarchy_attribute["parent_node_name"].unique()

            for v in df_hierarchy_attribute.to_dict(orient="records"):
                k = v["itemset_name"]

                if v["parent_node_name"]:
                    if v["attribute"] not in generalization_dict:
                        generalization_dict[v["attribute"]] = {}
                    generalization_dict[v["attribute"]][k] = v["parent_node_name"]

                # if v["hasChild"] is False:
                if k not in parent_nodes:
                    if v["attribute"] not in discretizations:
                        discretizations[v["attribute"]] = {}
                    discretizations[v["attribute"]][k] = {
                        "rels": v["rels"],
                        "vals": v["vals"],
                    }

        return generalization_dict, discretizations

    def getHierarchyAndDiscretizationSplits2(self, tree, verbose=False):
        # TODO: manage empy tree
        df_hierarchy = self.get_hierarchy_DF(tree)
        if df_hierarchy is None:
            return None, None
        generalization_dict = {}
        discretizations = {}
        print(verbose)
        for attribute, df_hierarchy_attribute in df_hierarchy.groupby("attribute"):
            if not df_hierarchy_attribute["itemset_name"].is_unique:
                import warnings

                if verbose:
                    print(
                        df_hierarchy_attribute[
                            df_hierarchy_attribute.duplicated(
                                ["itemset_name"], keep=False
                            )
                        ]
                    )

                warnings.warn(
                    f"Attribute {attribute} splitted in the same way multiple ways."
                )
            parent_nodes = df_hierarchy_attribute["parent_node_name"].unique()

            for v in df_hierarchy_attribute.to_dict(orient="records"):

                k = v["itemset_name"]
                if v["attribute"] not in discretizations:
                    discretizations[v["attribute"]] = {}
                discretizations[v["attribute"]][k] = {
                    "rels": v["rels"],
                    "vals": v["vals"],
                }

        return generalization_dict, discretizations

    """
    def getHierarchyAndDiscretizationSplitsAllAttributes(
        self, tree, min_max_values, continuous_attributes, verbose=False
    ):
        # TODO: manage empy tree
        df_hierarchy = self.get_hierarchy_DF(tree)
        if df_hierarchy is None:
            return None, None

        generalization_dict = {}
        discretizations = {}
        preserve_interval = {}
        for attribute, df_hierarchy_attribute in df_hierarchy.groupby("attribute"):

            if not df_hierarchy_attribute["itemset_name"].is_unique:
                import warnings

                if verbose:
                    print(
                        df_hierarchy_attribute[
                            df_hierarchy_attribute.duplicated(
                                ["itemset_name"], keep=False
                            )
                        ]
                    )

                warnings.warn(
                    f"Attribute {attribute} splitted in the same way multiple ways."
                )

            df_hierarchy_attribute = df_hierarchy_attribute.sort_values("level")

            intervals_dict = {}
            interval_list = []
            info_attribute_rels = df_hierarchy_attribute.to_dict(orient="records")
            parent_nodes = df_hierarchy_attribute["parent_node_name"].unique()
            interval_list_2 = []
            interval_list_new = []
            if attribute in continuous_attributes:

                for v in info_attribute_rels:
                    if attribute != v["attribute"]:
                        raise ValueError("attributes differ")

                    if attribute not in intervals_dict:
                        intervals_dict[attribute] = {}
                    interval = _get_intervals_from_relation(
                        v, list(min_max_values[attribute])
                    )
                    not_overlap = True
                    interval_list_new = []

                    print("E", "interval", interval)
                    for interval_old in interval_list_2:
                        overlap = get_overlap_intervals(interval, interval_old)
                        interval_point = get_overlap_intervals_point(
                            interval, interval_old
                        )

                        if overlap > 0:
                            print("E", "overlap2", overlap)
                            not_overlap = False
                            left, right = get_overlap_as_interval(
                                interval, interval_old
                            )
                            interval_list_new.append([left, right])
                        else:
                            if interval_point:
                                interval_list_new.append(interval_point[0])
                                interval_list_new.append(interval_point[1])
                            interval_list_new.append(interval_old)
                    if not_overlap:
                        interval_list_new.append(interval)

                    interval_list_2 = interval_list_new.copy()
                    print("E", "interval_list_2", interval_list_2, "\n")

                    intervals_dict[attribute][v["itemset_name"]] = interval
                    interval_list.extend(interval)

                print("E", "interval_list", interval_list)

                interval_list = get_edges_intervals(interval_list)
                for v in info_attribute_rels:
                    print("E", "v", v["itemset_name"])
                    to_add = True
                    print("E", "intervals_dict", intervals_dict)
                    if v["attribute"] in intervals_dict:

                        interval = tuple(
                            intervals_dict[v["attribute"]][v["itemset_name"]]
                        )
                        print("E", "interval", interval)
                        if interval not in interval_list:
                            print("E", "interval not in ", interval, interval_list)

                            to_add = False
                            (
                                parent_itemset_name,
                                parent_rels,
                                parent_vals,
                            ) = _convert_to_rels_and_name(
                                interval, min_max_values[attribute]
                            )

                            for interval_test in interval_list:
                                overlap = get_overlap_intervals(interval, interval_test)

                                print("E", "overlap", interval, interval_test)
                                if overlap > 0:
                                    (
                                        itemset_name_cut,
                                        rels_cut,
                                        vals_cut,
                                    ) = _convert_to_rels_and_name(
                                        interval_test, min_max_values[attribute]
                                    )

                                    if attribute not in discretizations:
                                        discretizations[attribute] = {}

                                    if attribute not in preserve_interval:
                                        preserve_interval[attribute] = set()

                                    preserve_interval[attribute].add(
                                        (v["itemset_name"])
                                    )
                                    print("E", "itemset_name_cut", itemset_name_cut)
                                    print("E", "added", v["itemset_name"])

                                    discretizations[attribute][itemset_name_cut] = {
                                        "rels": rels_cut,
                                        "vals": vals_cut,
                                    }
                                    if attribute not in generalization_dict:
                                        generalization_dict[attribute] = {}
                                    if (
                                        itemset_name_cut
                                        not in generalization_dict[attribute]
                                    ):
                                        generalization_dict[attribute][
                                            itemset_name_cut
                                        ] = parent_itemset_name
                                    else:
                                        gen_values = generalization_dict[attribute][
                                            itemset_name_cut
                                        ]
                                        if type(gen_values) != list:
                                            gen_values = [gen_values]
                                        gen_values.append(parent_itemset_name)
                                        generalization_dict[attribute][
                                            itemset_name_cut
                                        ] = gen_values

                    if to_add == True:
                        print("E", "to_add", to_add)

                        generalization_dict, discretizations = _add_info(
                            generalization_dict,
                            discretizations,
                            v["attribute"],
                            v["itemset_name"],
                            v["parent_node_name"],
                            v["vals"],
                            v["rels"],
                            parent_nodes,
                        )
                        if attribute not in preserve_interval:
                            preserve_interval[attribute] = set()
                        preserve_interval[attribute].add(v["itemset_name"])
                        print("E", "added", v["itemset_name"])
                print("E", "preserve_interval[attribute]", preserve_interval[attribute])
            else:
                for v in info_attribute_rels:
                    generalization_dict, discretizations = _add_info(
                        generalization_dict,
                        discretizations,
                        v["attribute"],
                        v["itemset_name"],
                        v["parent_node_name"],
                        v["vals"],
                        v["rels"],
                        parent_nodes,
                    )

        return generalization_dict, discretizations, preserve_interval
    """

    def getHierarchyAndDiscretizationSplitsAllAttributesConditioned(
        self, tree, verbose=False
    ):
        # TODO: manage empy tree
        df_hierarchy = self.get_hierarchy_DF(tree)
        if df_hierarchy is None:
            return None, None

        generalization_dict = {}
        discretizations = {}
        keep_info_parent_nodes = set()
        for attribute, df_hierarchy_attribute in df_hierarchy.groupby("attribute"):
            if not df_hierarchy_attribute["itemset_name"].is_unique:
                import warnings

                if verbose:
                    print(
                        df_hierarchy_attribute[
                            df_hierarchy_attribute.duplicated(
                                ["itemset_name"], keep=False
                            )
                        ]
                    )

                warnings.warn(
                    f"Attribute {attribute} splitted in the same way multiple ways."
                )
            parent_nodes = df_hierarchy_attribute["parent_node_name"].unique()
            for v in df_hierarchy_attribute.to_dict(orient="records"):
                k = v["itemset_name"]
                if v["parent_node_name"]:
                    if v["attribute"] not in generalization_dict:
                        generalization_dict[v["attribute"]] = {}
                    generalization_dict[v["attribute"]][k] = v["parent_node_name"]
                # if v["hasChild"] is False:
                if k not in parent_nodes:
                    if v["attribute"] not in discretizations:
                        discretizations[v["attribute"]] = {}
                    discretizations[v["attribute"]][k] = {
                        "rels": v["rels"],
                        "vals": v["vals"],
                        "parent": [],
                    }
                    for attribute, split_condition in v["parent_node_name_all_attrs"]:
                        if attribute != "root":
                            discretizations[v["attribute"]][k]["parent"].append(
                                (attribute, split_condition)
                            )
                            keep_info_parent_nodes.add(attribute)
        return generalization_dict, discretizations, keep_info_parent_nodes

    # TODO Merge ranking
    def visualizeTreeDiGraph(
        self,
        abbreviations={},
        rels={">=": "≥", "<=": "≤"},
        all_info=True,
        show_condition=False,
        ver2=False,
    ):

        from utils_print_tree import getTreeDiGraph

        # TODO
        self.get_hierarchy_DF(ver2=ver2)

        return getTreeDiGraph(
            self.tree,
            abbreviations=dict(abbreviations, **rels),
            type_criterion=self.criterion,
            metric_name=self.metric,
            all_info=all_info,
            show_condition=show_condition,
        )

    def get_internal_leaf_nodes(self, tree=None):
        tree = tree if tree else self.tree
        n_leaf, n_internal_nodes = [], []
        self.recurse_on_nodes(tree, n_leaf=n_leaf, n_internal_nodes=n_internal_nodes)
        if "root" in n_internal_nodes:
            n_internal_nodes.remove("root")
        if "root" in n_leaf:
            n_leaf.remove("root")
        return n_internal_nodes, n_leaf

    def get_number_internal_nodes(self):
        if self.number_internal_nodes is None:
            n_internal_nodes, n_leaf = self.get_internal_leaf_nodes()
            self.number_internal_nodes = len(n_internal_nodes)
            self.number_leaf_nodes = len(n_leaf)
        return self.number_internal_nodes

    def get_number_leaf_nodes(self):
        if self.number_leaf_nodes is None:
            n_internal_nodes, n_leaf = self.get_internal_leaf_nodes()
            self.number_internal_nodes = len(n_internal_nodes)
            self.number_leaf_nodes = len(n_leaf)
        return self.number_leaf_nodes

    def recurse_on_nodes(self, node, n_leaf=[], n_internal_nodes=[]):

        if node.children:
            n_internal_nodes.append(f"{node.attr}{node.rel}{node.val}")
            for child in node.children:
                self.recurse_on_nodes(
                    child, n_internal_nodes=n_internal_nodes, n_leaf=n_leaf
                )
        else:
            n_leaf.append(f"{node.attr}{node.rel}{node.val}")

    def get_keep_items_for_attributes(self):
        """
        For a given attribute, get the items to keep: the one that are associated with divergence (>0)
        """
        tree = self.tree
        keep_items = []
        self._iterate_and_get_divergent_node_relevant(tree, keep_items)

        return keep_items

    def _iterate_and_get_divergent_node_relevant(self, node, keep_items):

        """
        Recursively iterate over the tree to get the items with positive divergence. These are the ones associated with a divergence behavior.
        """

        has_children = True if node.children else False

        if node.item_name != None and node.metric > 0:
            keep_items.append(node.item_name)

        if has_children:
            for child in node.children:
                self._iterate_and_get_divergent_node_relevant(child, keep_items)


"""
    
    def selectBestSplit_ok(
        self,
        data_values,
        data_conf_matrix_values,
        dataset_len,
        min_support=0.1,
        type_criterion="sum_abs",
        gain=False,
        gain_parent = True, 
        verbose=False,
        parent_split_criterion_value = None, 
        parent_divergence = None,
        minimal_gain = None
    ):
        import numpy as np
        from utils_metric_tree import (
            isAttributeBinary,
            getSupport,
            getReciprocal,
        )

        best_split_info = None
        best_criterion = None

        def update_best_criterion(criterion, best_criterion, parent_split_criterion_value = None):
            if parent_split_criterion_value is not None and criterion == parent_split_criterion_value:
                return False
            if best_criterion is None:
                return True
            if type_criterion=="entropy":
                return criterion<best_criterion
            else:  #type_criterion=="KL" or 
                return criterion>best_criterion

        def check_gain_divergence(divergence_1, divergence_2, parent_divergence):
            if parent_divergence is None:
                # if not set, TODO remove this check
                return True
            elif (parent_divergence == divergence_1) and (parent_divergence == divergence_2):
                pr = False
                if pr:
                    print(parent_divergence , divergence_1, divergence_2)
                return False
            else:
                return True

        if getSupport(data_values, dataset_len) < min_support:
            return best_split_info
        for attr_id, attr_name in enumerate(self.attributes):
            if getSupport(data_values[:, attr_id], dataset_len) < min_support:
                continue
            
            attr_array = data_values[:, attr_id]
            if attr_name in self.discrete_attr:
                attr_vals = np.unique(attr_array)
                for val in attr_vals:
                    # Discrete splitting
                    id_eq = np.where(attr_array == val)
                    id_diff = np.where(attr_array != val)
                    part_1 = data_values[id_eq]
                    part_2 = data_values[id_diff]
                    sup_1 = getSupport(part_1, dataset_len)
                    sup_2 = getSupport(part_2, dataset_len)

                    if (sup_1 < min_support) or (sup_2 < min_support):
                        continue
                    
                    
                    from utils_metric_tree import evaluate_split_divergence
                    
                    criteria, divergence_1, divergence_2 = evaluate_split_divergence(data_conf_matrix_values[id_eq], data_conf_matrix_values[id_diff], \
                                                    self.root_metric, type_criterion = type_criterion, \
                                                        sup_n1=sup_1, sup_n2=sup_2, metric = self.metric, 
                                                        p_k_root = self.p_k_root, p_root = self.p_root)
                    
                    update = False
                    if gain:
                        metric_parent = getDivergenceMetric_np(
                            data_conf_matrix_values,
                            metric_baseline=self.root_metric,
                        )
                        print(criteria, metric_parent)
                        gain_v = abs(abs(criteria) - abs(metric_parent))
                        if gain_v > best_criterion:
                            update = True
                    else:
                        update =  update_best_criterion(criteria, best_criterion, parent_split_criterion_value)
                        if update:
                            update = check_gain_divergence(divergence_1, divergence_2, parent_divergence)
                        
       

                    if update:
                        best_criterion = criteria
                        isbinary = isAttributeBinary(attr_vals)
                        val2 = getReciprocal(attr_vals, val) if isbinary else val
                        rel2 = "=" if isbinary else "!="
                        best_split_info = {
                            "attr": attr_name,
                            "vals": (val, val2),
                            "rel": ("=", rel2),
                            "indexes": [
                                id_eq,
                                id_diff,
                            ],
                            "criterion": criteria,
                            "divergence" : [divergence_1, divergence_2]
                        }

                        
                    if isAttributeBinary(attr_vals):
                        break
            else:

                from copy import deepcopy

                index_sorted = np.argsort(data_values[:, attr_id])
                data_attr_sorted = data_values[index_sorted, attr_id]

                start_i = int(len(data_values) * min_support) - 1

                for i in range(start_i, len(data_values) - 1):
                    if data_attr_sorted[i] != data_attr_sorted[i + 1]:
                        val_1 = data_attr_sorted[i]
                        val_2 = data_attr_sorted[i + 1]

                        sup_1 = getSupport(data_attr_sorted[: i + 1], dataset_len)
                        sup_2 = getSupport(data_attr_sorted[i + 1 :], dataset_len)
                        if sup_1 < min_support:
                            continue
                        if sup_2 < min_support:
                            break
                        from utils_metric_tree import getDivergenceMetric_np

                        from utils_metric_tree import evaluate_split_divergence
                        if PRINT_VERBOSE:
                            print( attr_name, val_1, val_2)
                        criteria, divergence_1, divergence_2 = evaluate_split_divergence(data_conf_matrix_values[index_sorted[: i + 1]], data_conf_matrix_values[index_sorted[i + 1 :]], \
                                                    self.root_metric, type_criterion = type_criterion, sup_n1=sup_1, sup_n2=sup_2, metric = self.metric,\
                                                    p_k_root = self.p_k_root, p_root = self.p_root)



                        update = False
                        if gain:
                            metric_parent = getDivergenceMetric_np(
                                data_conf_matrix_values,
                                metric_baseline=self.root_metric,
                            )

                            gain_v = abs(abs(criteria) - abs(metric_parent))
                            if gain_v > best_criterion:
                                update = True
                        else:
                            update =  update_best_criterion(criteria, best_criterion, parent_split_criterion_value)
                            if update:
                                update = check_gain_divergence(divergence_1, divergence_2, parent_divergence)


                        if update:


                            best_criterion = criteria

                            best_split_info = {
                                "attr": attr_name,
                                "vals": (val_1, val_2),
                                "rel": ("<=", ">="),
                                "indexes": [
                                    index_sorted[: i + 1],
                                    index_sorted[i + 1 :],
                                ],
                                "criterion": criteria,
                                "divergence" : [divergence_1, divergence_2]

                            }
        if PRINT_VERBOSE:
            print('----------------------------------------')
        return best_split_info

    
"""


class Node:
    def __init__(
        self,
        metric,
        support,
        attr,
        val,
        rel,
        confusion_matrix=None,
        node_criterion=None,
        node_support_ratio=None,
        node_size=None,
        measure_node=None,
        info_values=None,
        node_split_value=None,
    ):
        self.metric = metric
        self.children = []
        self.support = support
        self.confusion_matrix = confusion_matrix
        self.attr = attr
        self.val = val
        self.rel = rel
        self.criterion_split = None
        self.entropy = None
        self.measure_node = measure_node
        self.criterion_value = node_criterion
        self.gain = None
        self.node_support_ratio = node_support_ratio
        self.node_size = node_size
        self.info_values = info_values
        self.item_name = None
        self.node_split_value = node_split_value

    def node_condition(self):
        if self.node_split_value is None:
            return "root", "", ""
        return (
            self.node_split_value[0],
            self.node_split_value[1],
            self.node_split_value[2],
        )

    def set_criterion_split(self, criterion_split):
        self.criterion_split = criterion_split

    def set_entropy(self, entropy):
        self.entropy = entropy

    def set_measure_node(self, measure_node):
        self.measure_node = measure_node

    def set_criterion_value(self, criterion_value):
        # measure_node
        self.criterion_value = criterion_value

    def set_gain(self, gain):
        self.gain = gain

    def set_item_name(self, item_name):
        self.item_name = item_name

    def __str__(self):
        return f"{self.attr} {self.rel} {self.val}"
