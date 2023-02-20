# Evalutate if merge in tree divergence class


from pickle import NONE


def getTreeDiGraph(
    tree,
    type_criterion=None,
    abbreviations={},
    metric_name=None,
    all_info=True,
    show_condition=False,
):
    id_node = 0

    list_nodes = []
    list_edges = []
    getNodeDiGraph(
        id_node,
        tree,
        list_nodes,
        list_edges,
        abbreviations=abbreviations,
        type_criterion=type_criterion,
        metric_name=metric_name,
        all_info=all_info,
        show_condition=show_condition,
    )

    import graphviz

    dot = graphviz.Digraph(
        comment="Tree discretization",
        node_attr={"shape": "rect", "nodesep": "0.1", "ranksep": "0.1"},
        # graph_attr={"nodesep": ".1", "ranksep": ".7"},
    )
    parents = set([x[0] for x in list_edges])
    for id_node, label_node in list_nodes:
        if id_node not in parents:
            dot.node(id_node, label_node, style="bold")
        else:
            dot.node(id_node, label_node)

    dot.edges(list_edges)
    return dot


def format_item_name(attribute, item_name, sep="="):
    if (item_name[0] == "<") or (item_name[0] == ">") or (item_name[0] == "="):
        sep = ""
    return f"{attribute}{sep}{item_name}"


def getNodeDiGraph(
    id_node,
    node,
    nodes,
    list_edges,
    node_id=0,
    parent_id=0,
    r=3,
    abbreviations={},
    type_criterion=None,
    metric_name=None,
    all_info=False,
    show_condition=False,
):
    is_root = False

    if show_condition:
        attr, rel, val = node.node_condition()
        item = f"{attr}{rel}{val}"
        if attr == "root":
            is_root = True
    else:
        item = f"Condition: {node.attr}{node.rel}{node.val}"
    item = abbreviateValue(item, abbreviations)
    print_vals = [item]

    if all_info:
        if node.item_name is not None:
            print_vals.append(format_item_name(node.attr, node.item_name))
        if node.criterion_split is not None:

            if type_criterion is not None:
                print_vals.append(f"{type_criterion} split:{node.criterion_split:.4f}")
            else:
                print_vals.append(f"{node.criterion_split:.2f}")
        if node.entropy is not None:
            print_vals.append(f"entropy: {node.entropy:.3f}")
        if node.criterion_value is not None:
            print_vals.append(f"Criterion: {node.criterion_value:.3f}")

    if all_info:
        if node.measure_node is not None:
            print_vals.append(
                f"{metric_name.replace('d_', '')}: {node.measure_node:.3f}"
            )
        if node.node_support_ratio is not None:
            print_vals.append(f"sup_ratio: {node.node_support_ratio:.3f}")

        if node.gain is not None:
            print_vals.append(f"gain: {node.gain:.10f}")

        print_vals.append(f"sup={node.support:.5f} Δ={node.metric:.5f}")  # .2f .2f

        if node.confusion_matrix is not None:
            print_vals.append(node.confusion_matrix)
        if node.info_values is not None:
            print_vals.append(node.info_values)
    else:
        if show_condition:
            print_vals.append(f"sup={node.support:.2f}")

            # if node.item_name is not None:
            #     print_vals.append(f"i: {format_item_name(node.attr, node.item_name)}")
            if is_root:
                print_vals.append(
                    f"{metric_name.replace('d_', '')}={node.measure_node:.2f}"
                )
            else:
                print_vals.append(f"Δ={node.metric:.2f}")
        else:
            print_vals.append(f"sup={node.support:.2f}")

            if node.item_name is not None:
                print_vals.append(format_item_name(node.attr, node.item_name))

            print_vals.append(
                f"{metric_name.replace('d_', '')}={node.measure_node:.3f} Δ={node.metric:.2f}"
            )

    print_vals = "\n".join(map(str, print_vals))
    # nodes.append((f"{id_node}", f'{item} \n {criterion} {impurity} \nSup={node.metric:.2f}  Δ={node.metric:.2f} \n {node.confusion_matrix}'))
    nodes.append((str(id_node), print_vals))

    if node.children:
        from copy import deepcopy

        parent_id = deepcopy(id_node)
        for child in node.children:
            id_node = max([int(x) for x, y in nodes])
            id_node = id_node + 1

            list_edges.append((f"{parent_id}", f"{id_node}"))
            getNodeDiGraph(
                id_node,
                child,
                nodes,
                list_edges,
                parent_id=parent_id,
                r=r,
                abbreviations=abbreviations,
                type_criterion=type_criterion,
                metric_name=metric_name,
                all_info=all_info,
                show_condition=show_condition,
            )


def abbreviateValue(value, abbreviations={}):
    for k, v in abbreviations.items():
        if k in value:
            value = value.replace(k, v)
    return value


def viz_tree(
    tree_discr,
    continuous_attributes=None,
    tree_outputdir=".",
    suffix="discr",
    saveFig=False,
    verbose=False,
):
    import os

    tree_discr.printDiscretizationTrees(round_v=3)
    dot_show = None
    if type(tree_discr.trees) is dict:
        dot = {}
        if continuous_attributes is None:
            attributes = list(tree_discr.trees.keys())
        else:
            attributes = continuous_attributes
        for attribute in attributes:
            if attribute in tree_discr.trees:
                dot[attribute] = tree_discr.trees[attribute].visualizeTreeDiGraph()
                if saveFig:
                    dot[attribute].render(
                        os.path.join(tree_outputdir, f"tree_{attribute}_{suffix}.pdf")
                    )
                dot_show = dot[attribute]
            else:
                if verbose:
                    print(f"Attribute {attribute} is not discretized")
    else:
        dot_show = tree_discr.trees.visualizeTreeDiGraph()
        if saveFig:

            dot_show.render(os.path.join(tree_outputdir, f"tree_{suffix}.pdf"))
