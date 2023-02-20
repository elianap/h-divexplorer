from utils_plot import get_predefined_color_labels, abbreviateValue
import json

def plot_experiment_results_varying_support(dataset_name = 'wine', min_support_tree = 0.1,
    type_criterion="divergence_criterion",
    metric = "d_fpr",
    output_dir = 'output_results_2',
    saveFig = True):

    # # Read performance results

    import os

    output_results = os.path.join(os.path.curdir, output_dir, 'performance')
    from pathlib import Path

    Path(output_results).mkdir(parents=True, exist_ok=True)

    conf_name = f"{dataset_name}_{metric}_{type_criterion}_{min_support_tree}"

    conf_name = f"{dataset_name}_{metric}_{type_criterion}_{min_support_tree}"



    with open(os.path.join(output_results, f'{conf_name}_time.json')) as json_file:
        result_time = json.load(json_file)

    with open(os.path.join(output_results, f'{conf_name}_fp.json')) as json_file:
        result_fp = json.load(json_file)

    with open(os.path.join(output_results, f'{conf_name}_div.json')) as json_file:
        result_maxdiv = json.load(json_file)    


    print(result_maxdiv)
    import os
    output_fig_dir = os.path.join(os.path.curdir, output_dir, "figures", "output_performance")

    if saveFig:
        

        from pathlib import Path

        Path(output_fig_dir).mkdir(parents=True, exist_ok=True)


    abbreviations = {"one_at_time":"indiv t.", \
                        "divergence_criterion":"g$\\Delta$", "entropy":"entr"}



    color_labels = get_predefined_color_labels(abbreviations)
    lines_style = {k:"-" for k in color_labels}
    lines_style.update({k:"--" for k in color_labels if( "base" in k and abbreviations["entropy"] in k)})
    lines_style.update({k:"-." for k in color_labels if( 'base' in k and abbreviations["divergence_criterion"] in k)})



    from utils_plot import plotDicts

    size_fig = (3,3)


    for info_i, results in [('time', result_time), (f"max_{metric}", result_maxdiv), ('FP', result_fp)]:

        info_plot = {}
        for sup in sorted(results.keys()):
            for type_gen in results[sup]:
                type_gen_str = abbreviateValue(f"{type_criterion}_{type_gen}", abbreviations)
                if type_gen_str not in info_plot:
                    info_plot[type_gen_str] = {}
                info_plot[type_gen_str][float(sup)] = results[sup][type_gen]

        figure_name = os.path.join(output_fig_dir, f"{dataset_name}_stree_{min_support_tree}_{metric}_{info_i}.pdf")
        
        for type_gen_str in info_plot:
            info_plot[type_gen_str] = dict(sorted(info_plot[type_gen_str].items()))


        title, ylabel = '', ''
        if info_i == 'time':
            title = 'Execution time'
            ylabel="Execution time $(seconds)$"

        elif info_i == f"max_{metric}":
            m = metric[2:].upper()
            
            ylabel=f"Max $\\Delta_{{{m}}}$"
            title=f"Highest $\\Delta_{{{m}}}$"

        elif info_i == 'FP':
            ylabel="#FP"
            title="#FP" 
            
        print(ylabel, info_i)
        print(figure_name)

        plotDicts(info_plot, marker=True, \
            title=title, sizeFig=size_fig,\
                    linestyle=lines_style, color_labels=color_labels, \
            xlabel="Minimum support s",  ylabel=ylabel, labelSize=10.2,\
            outside=True,  saveFig=saveFig, nameFig = figure_name, legendSize=10.2)
            



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="output_red",
        help="specify the name of the output folder",
    )

    parser.add_argument(
        "--no_show_figs",
        action="store_true",
        help="specify not_show_figures to vizualize the plots. The results are stored into the specified outpur dir.",
    )


    parser.add_argument(
        "--dataset_name",
        default="wine",
        help="specify the name of the dataset",
    )

    parser.add_argument(
        "--min_support_tree",
        type=float,
        default=0.1,
        help="specify the name of the dataset",
    )
    parser.add_argument(
        "--min_sup_divergences",
        type=float,
        nargs="*",
        default=[0.15, 0.2],
        help="specify the minimum support scores",
    )
    parser.add_argument(
        "--type_criterion",
        type=str,
        default="divergence_criterion",
        help="specify the split criterion",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='d_fpr',
        help="specify the metric",
    )
    parser.add_argument(
        "--min_instances",
        type=int,
        default=50,
        help="specify the number of minimum instances for statistical significance",
    )

    args = parser.parse_args()

    plot_experiment_results_varying_support(min_support_tree = args.min_support_tree,
    type_criterion=args.type_criterion,
    metric = args.metric,
    #saveFig = True,
        dataset_name = args.dataset_name,
        output_dir=args.name_output_dir
    )

# ! python plot_experiment_results_varying_support.py --dataset_name online_shoppers_intention --metric d_error