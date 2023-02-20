import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

np.random.seed(42)


def gaussian_error_data(n=10000, n_attributes=3, show_img=False):
    from scipy.stats import multivariate_normal
    import numpy as np

    np.random.seed(42)

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    np.random.seed(42)
    X = np.random.uniform(low=-5, high=5, size=(n, n_attributes))

    mean = np.arange(n_attributes)
    cov = np.ones(n_attributes)
    f_g = multivariate_normal(mean, cov)  # , [1, 1, 1])
    g = f_g.pdf(X)

    if show_img:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(X[:, 0], X[:, 1], g)
        plt.show()

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    g_sc = np.round_(scaler.fit_transform(g.reshape(-1, 1))[:, 0], 15)

    import string

    attributes = list(string.ascii_lowercase)[:n_attributes]

    classes = np.random.choice([0, 1], size=X.shape[0], p=[0.5, 0.5])
    opposed = 1 - classes
    values = np.vstack((classes, opposed)).T

    predicted_classes = [
        np.random.choice(values[i : i + 1][0], 1, p=[1 - g_sc[i], g_sc[i]])[0]
        for i in range(0, g_sc.shape[0])
    ]
    predicted_classes = np.asarray(predicted_classes)

    df_analysis = pd.DataFrame(
        np.hstack((X, classes.reshape(-1, 1), predicted_classes.reshape(-1, 1))),
        columns=attributes + ["true_class", "predicted_class"],
    ).round(5)
    return df_analysis, attributes, g, g_sc


def get_ranges_from_itemset(itemset):
    itemset_split = {}
    for item in itemset:

        if item == "":
            print(item)
            continue
        start_v, end_v = None, None
        s = item.split("=")
        attr, v = s[0], "=".join(s[1:])
        if v[0:2] == ">=":
            start_v = v[2:]
        elif v[0:2] == "<=":
            end_v = v[2:]
        else:
            v = v[1:-1]
            vals = v.split("-")
            if len(vals) == 2:
                start_v, end_v = vals[0], vals[1]
            else:
                start_v, end_v = vals[0], vals[-1]
                if vals[0] == "":
                    start_v = "-" + vals[1]
                if vals[-2] == "":
                    end_v = "-" + vals[-1]
        itemset_split[attr] = (
            float(start_v) if start_v != None else start_v,
            float(end_v) if end_v != None else end_v,
        )
    return itemset_split


def show_splits_axes(attr, ax, ranges, min_max_vals, h=2, eps=0, show_all_if_non=False):
    if attr not in ranges:
        if show_all_if_non:
            start_v, end_v = min_max_vals[attr][0], min_max_vals[attr][1]
            ax.axvspan(start_v, end_v, alpha=0.1)
            return ax
        else:
            print(f"Attribute {attr} not in the itemset")
            return ax
    start_v, end_v = ranges[attr]
    if start_v is None:
        start_v = min_max_vals[attr][0]
    if end_v is None:
        end_v = min_max_vals[attr][1]

    ax.axvspan(start_v, end_v, alpha=0.1)
    ax.hlines(y=h, xmin=start_v, xmax=end_v, linestyle="--")
    ax.scatter([start_v + eps], [h], marker="<", c="black")
    ax.scatter([end_v - eps], [h], marker=">", c="black")
    return ax


def add_splits_axes(
    attr,
    ax,
    tree_discr,
    min_max_vals,
    g,
    generalization_dict,
    cmap_type="tab10",
    show_span=True,
    show_lines=True,
    p=0.2,
    eps=0.1,
):

    splits = tree_discr.trees[attr].get_discretization_relations(
        apply_generalization=True, generalization_dict=generalization_dict
    )

    cmap = plt.get_cmap(cmap_type)
    levels = []
    for rel, value, level, rel_int, values_int in list(
        tree_discr.trees[attr]
        .printTreeHierarchy()[["rel", "value", "level", "rels_ret", "vals_ret"]]
        .values
    ):
        rel, value = list(rel)[0], list(value)[0]
        rel_int, values_int = rel_int[attr], values_int[attr]

        ax.axvline(x=value, color=cmap(level), label=level, linestyle="-")
        levels.append(level)
        # Min and max values
        start_v, end_v = min_max_vals[attr][0], min_max_vals[attr][1]

        y = max(g) - max(g) * level * p
        if len(values_int) == 1:
            if rel_int[0] == "<=":
                end_v = values_int[0]
            else:
                start_v = values_int[0]
        else:
            start_v = values_int[0]
            end_v = values_int[1]

        if show_span:
            ax.axvspan(start_v, end_v, alpha=0.1, color=cmap(level))
        if show_lines:
            ax.hlines(y=y, xmin=start_v, xmax=end_v, color=cmap(level), linestyle="--")
            ax.scatter([start_v + eps], [y], marker="<", color=cmap(level))
            ax.scatter([end_v - eps], [y], marker=">", color=cmap(level))
    return ax, cmap, levels


def plot_normal_attr(x, ax, g_sc, g_attr_i, plot_points=False):
    if plot_points:
        ax.scatter(x, g_sc, s=1, c="gray")
    ids = np.argsort(x)
    ax.plot(x[ids], g_attr_i[ids])
    return ax


def plot_attributes_split(
    df_vals,
    attributes,
    itemset,
    target_vals,
    g_attrs,
    min_max_vals,
    verbose=False,
    show_all_if_non=False,
    plot_points=False,
):

    ranges = get_ranges_from_itemset(itemset)
    if verbose:
        print("Ranges", ranges)
    eps = 0
    h = max(target_vals) + eps

    max_viz = 5
    if len(attributes) > max_viz:
        # Show at most max_viz attributes
        attributes = attributes[:max_viz]

    fig, axs = plt.subplots(1, len(attributes), figsize=(19, 4))
    for e, attribute in enumerate(attributes):
        axs[e] = plot_normal_attr(
            df_vals[attribute].values,
            axs[e],
            target_vals,
            g_attrs[e],
            plot_points=plot_points,
        )
        show_splits_axes(
            attribute,
            axs[e],
            ranges,
            min_max_vals,
            h,
            eps=eps,
            show_all_if_non=show_all_if_non,
        )
        axs[e].set_title(attribute)
    return fig
