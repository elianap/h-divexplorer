from sklearn import tree
from TreeDivergence import Node


def build_tree(sklearn_tree, feature_name, is_integer, verbose=False):
    def build_tree_recurse(idx, parent_idx, depth, is_left):
        spacing = 3

        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing

        # TODO
        num, den = sklearn_tree.value[0][0]
        ref = num / den

        dataset_len = sklearn_tree.n_node_samples[0]

        if idx == 0:
            attr = "root"
            val = ""
            rel = ""

            parent_size = sklearn_tree.n_node_samples[0]
            node_divergence = 0

        else:

            attr = feature_name
            val = sklearn_tree.threshold[parent_idx]
            rel = "<=" if is_left else ">"
            parent_size = sklearn_tree.n_node_samples[parent_idx]

            num, den = sklearn_tree.value[idx][0]
            value = num / den
            node_divergence = value - ref

            if is_integer:
                if is_left:
                    val = int(val)

                else:
                    val = int(val)
                    rel = ">"

        node_split_value = [
            attr,
            rel,
            val,
        ]

        node_size = sklearn_tree.n_node_samples[idx]
        support = sklearn_tree.n_node_samples[idx] / dataset_len

        confusion_matrix = None
        node_criterion = sklearn_tree.impurity[idx]
        node_support_ratio = node_size / parent_size

        if verbose:
            print(
                indent,
                attr,
                rel,
                "{1:.{0}f}".format(2, val) if type(val) != str else val,
                f"s: {support:.2f}",
                f"Impurity: {node_criterion:.2f}",
                f"Size_ratio: {node_support_ratio:.2f}",
                f"\tdiv {node_divergence:.2f}",
            )

        node_data = Node(
            node_divergence,
            support,
            attr,
            val,
            rel,
            confusion_matrix=confusion_matrix,
            node_criterion=node_criterion,
            node_support_ratio=node_support_ratio,
            node_split_value=node_split_value,
        )

        if sklearn_tree.feature[idx] != tree._tree.TREE_UNDEFINED:
            left = (sklearn_tree.children_left[idx],)
            right = sklearn_tree.children_right[idx]

            node_data.children = [
                build_tree_recurse(left, idx, depth + 1, True),
                build_tree_recurse(right, idx, depth + 1, False),
            ]

        return node_data

    return build_tree_recurse(0, None, 1, None)
