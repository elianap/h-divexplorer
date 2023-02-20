MARKERS = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "*",
    "8",
    "s",
    "p",
    "h",
    "P",
    ".",
    "d",
    "H",
    "X",
    "H",
    ",",
    4,
    5,
    6,
    7,
    "+",
    "x",
    "|",
    "_",
    "1",
    "2",
    "4",
]


def plotDicts(
    info_dicts,
    title="",
    xlabel="",
    ylabel="",
    marker=False,
    limit=None,
    nameFig="",
    colorMap="tab10",
    sizeFig=(4, 3),
    labelSize=10,
    markersize=4,
    linewidth=1.5,
    outside=False,
    titleLegend="",
    tickF=False,
    yscale="linear",
    legendSize=5,
    saveFig=False,
    show_figure=True,
    color_labels=None,
    xscale="linear",
    linestyle="-",
    borderpad=0.25,
    kformat=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    # plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100

    fig, ax = plt.subplots(figsize=sizeFig, dpi=100)

    if kformat:
        from matplotlib.ticker import FuncFormatter

        def kilos(x, pos):
            "The two args are the value and tick position"
            return f"{(x * 1e-3):.0f}k"

        formatter = FuncFormatter(kilos)

        ax.yaxis.set_major_formatter(formatter)

    m_i = 0

    if color_labels is None:
        if colorMap:

            colors = plt.get_cmap(colorMap).colors
            if len(colors) < len(info_dicts):
                i = 0
                cs = ["Pastel1", "Pastel2"]
                while (len(colors) < len(info_dicts)) and (i < len(cs)):
                    # TODO
                    colors = list(colors) + list(plt.get_cmap(cs[i]).colors)
                    i += 1
        else:
            import numpy as np

            colors = plt.cm.winter(np.linspace(0, 1, 20))
    else:
        colors = [color_labels[label] for label in info_dicts]

    keys = list(info_dicts.keys())
    if linestyle == "-":
        linestyle = {k: "-" for k in keys}

    for e, (label_name, info_dict) in enumerate(info_dicts.items()):
        if marker:
            ax.plot(
                list(info_dict.keys()),
                list(info_dict.values()),
                label=label_name,
                marker=MARKERS[m_i],
                linewidth=linewidth,
                markersize=markersize,
                color=colors[e],
                linestyle=linestyle[label_name],
            )
            m_i = m_i + 1
        else:
            ax.plot(
                list(info_dict.keys()),
                list(info_dict.values()),
                label=label_name,
                color=colors[e],
                linestyle=linestyle[label_name],
            )
    import cycler

    if colorMap:
        plt.rcParams["axes.prop_cycle"] = cycler.cycler(
            color=plt.get_cmap(colorMap).colors
        )
    else:
        color = plt.cm.winter(np.linspace(0, 1, 10))
        plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", color)

    if limit is not None:
        plt.ylim(top=limit[1], bottom=limit[0])

    if not kformat and tickF:
        print("bbb")
        xt = list(info_dict.keys())
        plt.xticks(
            [xt[i] for i in range(0, len(xt)) if i == 0 or xt[i] * 100 % 5 == 0],
            fontsize=labelSize,
        )

    plt.xlabel(xlabel, fontsize=labelSize)
    plt.ylabel(ylabel, fontsize=labelSize)
    plt.title(title, fontsize=labelSize)

    plt.xscale(xscale)
    if not kformat:
        plt.yscale(yscale)
    if outside:
        plt.legend(
            prop={"size": legendSize},
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            title=titleLegend,
            fontsize=5,
            title_fontsize=5,
        )
    else:
        plt.legend(
            prop={"size": legendSize},
            title=titleLegend,
            fontsize=legendSize,
            title_fontsize=legendSize,
            borderpad=borderpad,
        )
    if saveFig:
        fig.tight_layout()
        plt.savefig(nameFig, bbox_inches="tight")
    if show_figure:
        plt.show()
        plt.close()


def two_plots_shared_labels(
    info_dicts,
    info_dicts2,
    figure_name="fig",
    label_1="1",
    label_2="2",
    ylabel="",
    save_fig=False,
    sizeFig=(6, 4),
    markersize=3,
    linewidth=1,
    xlabel="Minimum support s",
    fontsize=12,
    log_scale=True,
    title="",
    outside=False,
    color_labels=None,
    label_1_short="a",
    label_2_short="b",
    xlog_scale=False,
    sharex=False,
    use_shared_labels=True,
    borderpad=0.25,
):

    import matplotlib.pyplot as plt

    m_i = 0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, sharex=sharex, figsize=sizeFig, dpi=100
    )

    if color_labels is None:
        colors = plt.get_cmap("tab10").colors
        if len(colors) < len(info_dicts):
            i = 0
            cs = ["Pastel1", "Pastel2"]
            while (len(colors) < len(info_dicts)) and (i < len(cs)):
                # TODO
                colors = list(colors) + list(plt.get_cmap(cs[i]).colors)
                i += 1
    else:
        colors = [color_labels[label] for label in info_dicts]

    for e, (label, info_dict) in enumerate(info_dicts.items()):
        if info_dicts[label]:
            ax1.plot(
                list(info_dict.keys()),
                list(info_dict.values()),
                label=label,
                marker=MARKERS[m_i],
                linewidth=linewidth,
                markersize=markersize,
                color=colors[e],
            )
        if info_dicts2[label]:
            ax2.plot(
                list(info_dicts2[label].keys()),
                list(info_dicts2[label].values()),
                label=label,
                marker=MARKERS[m_i],
                linewidth=linewidth,
                markersize=markersize,
                color=colors[e],
            )
        m_i = m_i + 1

    # plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100

    xlabel1 = f"{xlabel}\n\n({label_1_short})" if label_1_short else xlabel
    xlabel2 = f"{xlabel}\n\n({label_2_short})" if label_2_short else xlabel
    ax1.set_xlabel(xlabel1, fontsize=fontsize)
    ax2.set_xlabel(xlabel2, fontsize=fontsize)

    if log_scale:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    if xlog_scale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")

    if label_1:
        ax1.set_title(label_1, fontsize=fontsize)
    if label_2:
        ax2.set_title(label_2, fontsize=fontsize)

    ax1.set_ylabel(ylabel, fontsize=fontsize)

    fig.tight_layout(pad=0.5)
    if outside:
        plt.legend(
            prop={"size": fontsize},
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            title=title,
            fontsize=fontsize,
            title_fontsize=fontsize,
            borderpad=borderpad,
        )
    else:
        if use_shared_labels:
            ax2.legend(title=title, fontsize=fontsize, borderpad=borderpad)
        else:
            ax1.legend(title=title, fontsize=fontsize, borderpad=borderpad)
            ax2.legend(title=title, fontsize=fontsize, borderpad=borderpad)

    if save_fig:

        plt.savefig(figure_name, bbox_inches="tight")

    plt.show()
    plt.close()


def two_plots(
    info_dicts,
    info_dicts2,
    figure_name="fig",
    label_1="1",
    label_2="2",
    ylabel="",
    save_fig=False,
    sizeFig=(6, 4),
    markersize=3,
    linewidth=1,
    xlabel="Minimum support s",
    fontsize=12,
    log_scale=True,
    xlog_scale=False,
    title="Exp",
    outside=False,
    color_labels=None,
    label_1_short="a",
    label_2_short="b",
    x_ticks=False,
    legend_size=8,
    ylimit=None,
    sharex=False,
    bbox_to_anchor=(1.04, 1),
    pad=0.5,
    not_use_marker_list=None,
    linestyle="-",
):

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=sharex, sharey=True, figsize=sizeFig, dpi=100
    )

    keys = set(list(info_dicts.keys()) + list(info_dicts2.keys()))

    if color_labels is None:
        colors = plt.get_cmap("tab10").colors
        if len(colors) < (len(info_dicts) + len(info_dicts2)):
            # TODO
            colors = list(colors) + list(plt.get_cmap("Pastel1").colors)
            # print(len(colors))
    else:
        colors = [color_labels[label] for label in info_dicts]
        colors = colors + [color_labels[label] for label in info_dicts2]

    if linestyle == "-":
        linestyle = {k: "-" for k in keys}

    plot_labels = []
    labels_shared = set(info_dicts.keys()).intersection(info_dicts2.keys())
    lc = len(colors)
    markers_shared = {}
    color_shared = {}

    e = len(info_dicts)
    m_i = 0
    for e, (label, info_dict) in enumerate(info_dicts.items()):
        m_i = m_i + 1
        id_i = e % lc

        use_marker = True
        if not_use_marker_list is not None:
            if label in not_use_marker_list:
                marker = None
                use_marker = False
        if use_marker:
            marker = MARKERS[m_i]

        p1 = ax1.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            color=colors[id_i],
            linestyle=linestyle[label],
        )
        if label in labels_shared:
            markers_shared[label] = marker
            color_shared[label] = colors[id_i]
        plot_labels.append(label)

    e = len(info_dict)
    for e2, (label, info_dict) in enumerate(info_dicts2.items()):
        m_i = m_i + 1
        id_i = (e2 + e) % lc

        use_marker = True
        if not_use_marker_list is not None:
            if label in not_use_marker_list:
                marker = None
                use_marker = False
        if use_marker:
            if label not in labels_shared:
                marker = MARKERS[m_i]
            else:
                marker = markers_shared[label]

        p2 = ax2.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label if label not in labels_shared else None,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            linestyle=linestyle[label],
            color=colors[id_i] if label not in labels_shared else color_shared[label],
        )
        if label not in labels_shared:
            plot_labels.append(label)

    # plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    xlabel1 = f"{xlabel}\n\n({label_1_short})" if label_1_short else xlabel
    xlabel2 = f"{xlabel}\n\n({label_2_short})" if label_2_short else xlabel
    ax1.set_xlabel(xlabel1, fontsize=fontsize)
    ax2.set_xlabel(xlabel2, fontsize=fontsize)

    if log_scale:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    if xlog_scale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    if ylimit:
        ax1.set_ylim(ylimit)
        ax2.set_ylim(ylimit)
    ax1.set_title(label_1, fontsize=fontsize)
    ax2.set_title(label_2, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)

    fig.tight_layout(pad=pad)
    # if outside:
    #     plt.legend(
    #         prop={"size": fontsize},
    #         bbox_to_anchor=(1.04, 1),
    #         loc="upper left",
    #         title=title,
    #         fontsize=fontsize,
    #         title_fontsize=fontsize,
    #     )
    # else:
    #     ax2.legend(title=title, fontsize=fontsize)

    l = fig.legend(
        # [p1, p2],
        # labels=plot_labels,
        prop={"size": legend_size},
        bbox_to_anchor=(bbox_to_anchor),
        loc="upper left",
        title=title,
        fontsize=legend_size,
        title_fontsize=fontsize,
    )

    if save_fig:

        plt.savefig(figure_name, bbox_inches="tight")

    plt.show()
    plt.close()


def three_plots(
    info_dicts,
    info_dicts2,
    info_dicts3,
    figure_name="fig",
    labels_name=["1", "2", "3"],
    ylabel="",
    save_fig=False,
    sizeFig=(6, 4),
    markersize=3,
    linewidth=1,
    xlabel="Minimum support s",
    fontsize=12,
    log_scale=True,
    title=None,
    outside=False,
    color_labels=None,
    labels_short=["a", "b", "c"],
    pad=0.2,
    sharex=False,
    linestyle="-",
):

    import matplotlib.pyplot as plt

    m_i = 0

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, sharex=sharex, sharey=True, figsize=sizeFig, dpi=100
    )
    keys = set(
        list(info_dicts.keys()) + list(info_dicts2.keys()) + list(info_dicts3.keys())
    )
    if color_labels is None:
        colors = plt.get_cmap("tab10").colors
        if len(colors) < (len(keys)):
            # TODO
            colors = list(colors) + list(plt.get_cmap("Pastel1").colors)

        if len(colors) < len(keys):
            raise ValueError("TODO")
        color_labels = dict(zip(keys, colors[0 : len(keys)]))
    if linestyle == "-":
        linestyle = {k: "-" for k in keys}

    labels_shared = set(info_dicts.keys()).intersection(info_dicts2.keys())
    markers_shared = {}
    for label, info_dict in info_dicts.items():

        m_i = m_i + 1
        p1 = ax1.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label,
            marker=MARKERS[m_i],
            linewidth=linewidth,
            markersize=markersize,
            color=color_labels[label],
            linestyle=linestyle[label],
        )

        if label in labels_shared:
            markers_shared[label] = MARKERS[m_i]
    e = len(info_dicts)

    for label, info_dict in info_dicts2.items():
        m_i = m_i + 1
        p2 = ax2.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label if label not in labels_shared else None,
            marker=MARKERS[m_i]
            if label not in labels_shared
            else markers_shared[label],
            linewidth=linewidth,
            markersize=markersize,
            color=color_labels[label],
            linestyle=linestyle[label],
        )
        if label in labels_shared:
            markers_shared[label] = MARKERS[m_i]

    e = len(info_dicts) + len(info_dicts3)
    for label, info_dict in info_dicts3.items():

        m_i = m_i + 1
        p3 = ax3.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label if label not in labels_shared else None,
            marker=MARKERS[m_i]
            if label not in labels_shared
            else markers_shared[label],
            linewidth=linewidth,
            markersize=markersize if "unif" not in label else 3,
            color=color_labels[label],
            linestyle=linestyle[label],
        )

        if label in labels_shared:
            markers_shared[label] = MARKERS[m_i]

    # plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    for e, axis in enumerate([ax1, ax2, ax3]):
        axis.set_xlabel(f"{xlabel}\n\n({labels_short[e]})", fontsize=fontsize)
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(labels_name[e], fontsize=fontsize)
    if ylabel:
        ax1.set_ylabel(ylabel, fontsize=fontsize)

    fig.tight_layout(pad=pad)
    # if outside:

    fig.legend(
        # [p1, p2],
        # labels=plot_labels,
        prop={"size": fontsize},
        bbox_to_anchor=(1.01, 1),  # 1.04
        loc="upper left",
        title=title,
        fontsize=fontsize,
        title_fontsize=fontsize,
    )

    # fig.legend(
    #     prop={"size": fontsize},
    #     title=title,
    #     fontsize=fontsize,
    #     title_fontsize=fontsize,
    # )

    if save_fig:

        plt.savefig(figure_name, bbox_inches="tight", dpi=1000)

    plt.show()
    plt.close()


def three_plots_std(
    info_dicts,
    info_dicts2,
    info_dicts3,
    figure_name="fig",
    labels_name=["1", "2", "3"],
    ylabel="",
    save_fig=False,
    sizeFig=(6, 4),
    markersize=3,
    linewidth=1,
    xlabel="Minimum support s",
    fontsize=12,
    log_scale=True,
    title=None,
    outside=False,
    color_labels=None,
    labels_short=["a", "b", "c"],
    pad=0.2,
    sharex=False,
    linestyle="-",
    errorbar=None,
):

    import matplotlib.pyplot as plt

    m_i = 0

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, sharex=sharex, sharey=True, figsize=sizeFig, dpi=100
    )
    keys = set(
        list(info_dicts.keys()) + list(info_dicts2.keys()) + list(info_dicts3.keys())
    )
    if color_labels is None:
        colors = plt.get_cmap("tab10").colors
        if len(colors) < (len(keys)):
            # TODO
            colors = list(colors) + list(plt.get_cmap("Pastel1").colors)

        if len(colors) < len(keys):
            raise ValueError("TODO")
        color_labels = dict(zip(keys, colors[0 : len(keys)]))
    if linestyle == "-":
        linestyle = {k: "-" for k in keys}

    labels_shared = set(info_dicts.keys()).intersection(info_dicts2.keys())
    markers_shared = {}
    for label, info_dict in info_dicts.items():

        m_i = m_i + 1
        p1 = ax1.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label,
            marker=MARKERS[m_i],
            linewidth=linewidth,
            markersize=markersize,
            color=color_labels[label],
            linestyle=linestyle[label],
        )

        if label in labels_shared:
            markers_shared[label] = MARKERS[m_i]
    e = len(info_dicts)

    for label, info_dict in info_dicts2.items():
        m_i = m_i + 1
        if errorbar is None or label not in errorbar:
            p2 = ax2.plot(
                list(info_dict.keys()),
                list(info_dict.values()),
                label=label if label not in labels_shared else None,
                marker=MARKERS[m_i]
                if label not in labels_shared
                else markers_shared[label],
                linewidth=linewidth,
                markersize=markersize,
                color=color_labels[label],
                linestyle=linestyle[label],
            )
        else:
            p2 = ax2.errorbar(
                list(info_dict.keys()),
                list(info_dict.values()),
                list(errorbar[label].values()),
                label=label if label not in labels_shared else None,
                marker=MARKERS[m_i]
                if label not in labels_shared
                else markers_shared[label],
                linewidth=linewidth,
                markersize=markersize,
                color=color_labels[label],
                linestyle=linestyle[label],
            )

        if label in labels_shared:
            markers_shared[label] = MARKERS[m_i]

    e = len(info_dicts) + len(info_dicts3)
    for label, info_dict in info_dicts3.items():

        m_i = m_i + 1
        if errorbar is None or label not in errorbar:
            p3 = ax3.plot(
                list(info_dict.keys()),
                list(info_dict.values()),
                label=label if label not in labels_shared else None,
                marker=MARKERS[m_i]
                if label not in labels_shared
                else markers_shared[label],
                linewidth=linewidth,
                markersize=markersize if "unif" not in label else 5,
                color=color_labels[label],
                linestyle=linestyle[label],
            )
        else:
            p3 = ax3.errorbar(
                list(info_dict.keys()),
                list(info_dict.values()),
                list(errorbar[label].values()),
                label=label if label not in labels_shared else None,
                marker=MARKERS[m_i]
                if label not in labels_shared
                else markers_shared[label],
                linewidth=linewidth,
                markersize=markersize if "unif" not in label else 5,
                color=color_labels[label],
                linestyle=linestyle[label],
            )

        if label in labels_shared:
            markers_shared[label] = MARKERS[m_i]

    # plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    for e, axis in enumerate([ax1, ax2, ax3]):
        axis.set_xlabel(f"{xlabel}\n\n({labels_short[e]})", fontsize=fontsize)
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(labels_name[e], fontsize=fontsize)
    if ylabel:
        ax1.set_ylabel(ylabel, fontsize=fontsize)

    fig.tight_layout(pad=pad)
    # if outside:

    fig.legend(
        # [p1, p2],
        # labels=plot_labels,
        prop={"size": fontsize},
        bbox_to_anchor=(1.01, 1),  # 1.04
        loc="upper left",
        title=title,
        fontsize=fontsize,
        title_fontsize=fontsize,
    )

    # fig.legend(
    #     prop={"size": fontsize},
    #     title=title,
    #     fontsize=fontsize,
    #     title_fontsize=fontsize,
    # )

    if save_fig:

        plt.savefig(figure_name, bbox_inches="tight", dpi=1000)

    plt.show()
    plt.close()


def two_plots_shared_labels_distinct_axis(
    info_dicts,
    info_dicts2,
    figure_name="fig",
    label_1="1",
    label_2="2",
    ylabel_1="",
    ylabel_2="",
    save_fig=False,
    sizeFig=(6, 4),
    markersize=3,
    linewidth=1,
    xlabel="Minimum support s",
    fontsize=12,
    log_scale_1=True,
    log_scale_2=True,
    x_log_scale_1=False,
    x_log_scale_2=False,
    title="",
    outside=False,
    color_labels=None,
    label_1_short="a",
    label_2_short="b",
    legend_size=8,
    y1_limit=None,
    y2_limit=None,
    bbox_to_anchor=(1.04, 1),
):

    import matplotlib.pyplot as plt

    m_i = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=sizeFig, dpi=100)  # , sharey=True)

    if color_labels is None:
        colors = plt.get_cmap("tab10").colors
        if len(colors) < len(info_dicts):
            # TODO
            colors = list(colors) + list(plt.get_cmap("Pastel1").colors)
    else:
        colors = [color_labels[label] for label in info_dicts]

    for e, (label, info_dict) in enumerate(info_dicts.items()):
        ax1.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label,
            marker=MARKERS[m_i],
            linewidth=linewidth,
            markersize=markersize,
            color=colors[e],
        )
        ax2.plot(
            list(info_dicts2[label].keys()),
            list(info_dicts2[label].values()),
            label=label,
            marker=MARKERS[m_i],
            linewidth=linewidth,
            markersize=markersize,
            color=colors[e],
        )
        m_i = m_i + 1

    # plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    xlabel1 = f"{xlabel}\n\n({label_1_short})" if label_1_short else xlabel
    xlabel2 = f"{xlabel}\n\n({label_2_short})" if label_2_short else xlabel
    ax1.set_xlabel(xlabel1, fontsize=fontsize)
    ax2.set_xlabel(xlabel2, fontsize=fontsize)

    if log_scale_1:
        ax1.set_yscale("log")
    if log_scale_2:
        ax2.set_yscale("log")
    if x_log_scale_1:
        ax1.set_xscale("log")
    if x_log_scale_2:
        ax2.set_xscale("log")
    if y1_limit:
        ax1.set_ylim(y1_limit)
    if y2_limit:
        ax2.set_ylim(y2_limit)
    ax1.set_title(label_1, fontsize=fontsize)
    ax2.set_title(label_2, fontsize=fontsize)
    ax1.set_ylabel(ylabel_1, fontsize=fontsize)
    ax2.set_ylabel(ylabel_2, fontsize=fontsize)

    fig.tight_layout(pad=0.5)
    if outside:
        plt.legend(
            prop={"size": legend_size},
            bbox_to_anchor=bbox_to_anchor,
            loc="upper left",
            title=title,
            fontsize=legend_size,
            title_fontsize=fontsize,
        )
    else:
        ax2.legend(title=title, fontsize=fontsize)

    if save_fig:

        plt.savefig(figure_name, bbox_inches="tight")

    plt.show()
    plt.close()


def two_plots_v2(
    info_dicts,
    info_dicts2,
    figure_name="fig",
    label_1="1",
    label_2="2",
    ylabel="",
    save_fig=False,
    sizeFig=(6, 4),
    markersize=3,
    linewidth=1,
    xlabel="Minimum support s",
    fontsize=12,
    log_scale=True,
    xlog_scale=False,
    title="Exp",
    outside=False,
    color_labels=None,
    label_1_short="a",
    label_2_short="b",
    x_ticks=False,
    legend_size=8,
    ylimit=None,
    sharex=False,
    bbox_to_anchor=(1.04, 1),
    pad=0.5,
    output_order=None,
):

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=sharex, sharey=True, figsize=sizeFig, dpi=100
    )
    keys = list(info_dicts.keys()) + list(info_dicts2.keys())
    if color_labels is None:
        colors = plt.get_cmap("tab10").colors
        if len(colors) < (len(info_dicts) + len(info_dicts2)):
            # TODO
            colors = list(colors) + list(plt.get_cmap("Pastel1").colors)

            color_labels = {k: colors[e] for e, k in enumerate(keys)}
            # print(len(colors))
    marker_map = {k: MARKERS[e] for e, k in enumerate(keys)}
    plot_labels = []
    labels_shared = set(info_dicts.keys()).intersection(info_dicts2.keys())
    lc = len(keys)
    markers_shared = {}
    color_shared = {}

    e = len(info_dicts)
    m_i = 0
    for e, (label, info_dict) in enumerate(info_dicts.items()):
        m_i = m_i + 1
        id_i = e % lc
        p1 = ax1.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label,
            marker=marker_map[label],
            linewidth=linewidth,
            markersize=markersize,
            color=color_labels[label],
        )
        plot_labels.append(label)

    e = len(info_dict)
    for e2, (label, info_dict) in enumerate(info_dicts2.items()):
        m_i = m_i + 1
        id_i = (e2 + e) % lc
        p2 = ax2.plot(
            list(info_dict.keys()),
            list(info_dict.values()),
            label=label if label not in labels_shared else None,
            marker=marker_map[label],
            linewidth=linewidth,
            markersize=markersize,
            color=color_labels[label],
        )

    xlabel1 = f"{xlabel}\n\n({label_1_short})" if label_1_short else xlabel
    xlabel2 = f"{xlabel}\n\n({label_2_short})" if label_2_short else xlabel
    ax1.set_xlabel(xlabel1, fontsize=fontsize)
    ax2.set_xlabel(xlabel2, fontsize=fontsize)

    if log_scale:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    if xlog_scale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    if ylimit:
        ax1.set_ylim(ylimit)
        ax2.set_ylim(ylimit)
    ax1.set_title(label_1, fontsize=fontsize)
    ax2.set_title(label_2, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)

    fig.tight_layout(pad=pad)
    handles_1, labels_1 = ax1.get_legend_handles_labels()

    (
        handles_2,
        labels_2,
    ) = ax2.get_legend_handles_labels()  # plt.gca().get_legend_handles_labels()

    if output_order:
        legend_handle = []
        legend_label = []
        for l in output_order:
            if l in labels_1:
                i = labels_1.index(l)
                legend_label.append(l)
                legend_handle.append(handles_1[i])
            elif l in labels_2:
                i = labels_2.index(l)
                legend_label.append(l)
                legend_handle.append(handles_2[i])
            else:
                raise ValueError()
        l = fig.legend(
            legend_handle,
            legend_label,
            prop={"size": legend_size},
            bbox_to_anchor=(bbox_to_anchor),
            loc="upper left",
            title=title,
            fontsize=legend_size,
            title_fontsize=fontsize,
        )
        print(legend_label)

    else:
        l = fig.legend(
            # [handles[idx] for idx in order],[labels[idx] for idx in order]
            # [p1, p2],
            # labels=plot_labels,
            prop={"size": legend_size},
            bbox_to_anchor=(bbox_to_anchor),
            loc="upper left",
            title=title,
            fontsize=legend_size,
            title_fontsize=fontsize,
        )

    if save_fig:

        plt.savefig(figure_name, bbox_inches="tight")

    plt.show()
    plt.close()










######################



def abbreviateValue(value, abbreviations={}):
    for k, v in abbreviations.items():
        if k in value:
            
            value = value.replace(k, v)
    #TODO
    if value[0:2] not in ["q_", "u_"]:
        value = value.replace("_", " ")
    return value
    
def abbreviate_dict_value(input_dict, abbreviations):
    
    conv ={}
    for k1, dict_i in input_dict.items():
        conv[k1] = { abbreviateValue(k, abbreviations): d for k, d in dict_i.items()}
    return conv


def get_predefined_color_labels(abbreviations = {}):
    color_labels = {}
        
    color_labels[abbreviateValue(f'entropy_base', abbreviations)]="#7fcc7f"
    color_labels[abbreviateValue(f'divergence_criterion_base', abbreviations)]="#009900"

    color_labels[abbreviateValue(f'entropy_generalized', abbreviations)]="mediumblue"
    color_labels[abbreviateValue(f'divergence_criterion_generalized', abbreviations)]="orangered"


    color_labels[abbreviateValue(f'entropy_base_pruned', abbreviations)]="yellow"
    color_labels[abbreviateValue(f'divergence_criterion_base_pruned', abbreviations)]="#C179EE"

    color_labels[abbreviateValue(f'entropy_generalized_pruned', abbreviations)]="gray"
    color_labels[abbreviateValue(f'divergence_criterion_generalized_pruned', abbreviations)]="#C01FB1"

    return color_labels

