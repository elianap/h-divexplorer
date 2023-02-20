#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings("ignore")


import numpy as np
import os
import pandas as pd
from divexplorer_generalized.FP_Divergence import FP_Divergence

DATASET_DIRECTORY = os.path.join(os.path.curdir, "datasets")

# # Import data

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



def run_pruning_experiemnt(dataset_name = 'wine', min_support_tree = 0.1,
    min_sup_divergences = [0.1, 0.15, 0.2],
    type_criterion="divergence_criterion",
    type_experiment = "one_at_time",
    metric = "d_fpr",
    save = True,
    output_dir = 'output_results_2',
    saveFig = True,
    dataset_dir = DATASET_DIRECTORY):

    print(dataset_name)
    print(min_sup_divergences)
    print(output_dir)

    if dataset_name == 'wine':
        from import_process_dataset import import_process_wine, train_classifier_kv

        df, class_map, continuous_attributes = import_process_wine()
        # # Train and predict with RF classifier

        df = train_classifier_kv(df)
    elif dataset_name== "compas":
        from import_process_dataset import import_compas

        df, class_map, continuous_attributes = import_compas()

    elif dataset_name== "adult":
        from import_process_dataset import import_process_adult

        df, class_map, continuous_attributes = import_process_adult()

        from sklearn.preprocessing import LabelEncoder


        attributes = df.columns.drop("class")
        X = df[attributes].copy()
        y = df["class"].copy()

        encoders = {}
        for column in attributes:
            if df.dtypes[column] == np.object:
                print(column)
                le = LabelEncoder()
                X[column] = le.fit_transform(df[column])
                encoders[column] = le

        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_predict
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True
                    )  # Added to fix the random state  #Added shuffle=True for new version sklearn, Value Error
            
        y_predicted = cross_val_predict(clf, X, y.values, cv=cv)
        df["predicted"] = y_predicted
    else:
        raise ValueError()

    # # Tree divergence

    true_class_name = "class"
    pred_class_name = "predicted"
    cols_c = [true_class_name, pred_class_name]





    df_analyze = df.copy()

    from tree_discretization import TreeDiscretization

    tree_discr = TreeDiscretization()

    # ## Extract tree
    generalization_dict, discretizations = tree_discr.get_tree_discretization(
        df_analyze,
        type_splitting=type_experiment,
        min_support=min_support_tree,
        metric=metric,
        class_map=class_map,
        continuous_attributes=list(continuous_attributes),
        class_and_pred_names=cols_c,
        storeTree=True,
        type_criterion=type_criterion, 
        #minimal_gain = 0.0015
    )



    # # Extract patterns


    out_support = {}
    out_time = {}
    out_fp = {}



    from utils_extract_divergence_generalized import (
        extract_divergence_generalized,
    )
    import time

    for apply_generalization in [False, True]:
        type_gen = 'generalized' if apply_generalization else 'base'
        print(type_gen)
        for keep in [True, False]:
            if keep:
                keep_items = tree_discr.get_keep_items_associated_with_divergence()
                keep_str = "_pruned"
            else:
                keep_items = None
                keep_str = ""
            print(keep_str)
            for min_sup_divergence in min_sup_divergences:
                print(min_sup_divergence, end = " ")
                s_time = time.time()
                FP_fm = extract_divergence_generalized(
                    df_analyze,
                    discretizations,
                    generalization_dict,
                    continuous_attributes,
                    min_sup_divergence=min_sup_divergence,
                    apply_generalization=apply_generalization,
                    true_class_name=true_class_name,
                    predicted_class_name=pred_class_name,
                    class_map=class_map,
                    metrics_divergence = [metric],
                    FPM_type="fpgrowth",
                    save_in_progress = False, 
                    keep_only_positive_divergent_items=keep_items
                )

                key = type_gen + keep_str
                
                out_time.setdefault(min_sup_divergence, {})[key] = time.time()-s_time

                print(f"({(time.time()-s_time):.2f})")

                fp_divergence_i = FP_Divergence(FP_fm, metric=metric)

                most_divergent = (
                    fp_divergence_i.getDivergence(th_redundancy=0)
                    .sort_values(
                        [fp_divergence_i.metric, fp_divergence_i.t_value_col], ascending=False
                    )
                    .head(1)
                )
                out_support.setdefault(min_sup_divergence, {})[key] = most_divergent

                out_fp.setdefault(min_sup_divergence, {})[key] = len(FP_fm)



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



    size_fig = (3,3)
    from utils_plot import plotDicts

    out_support_max = {}


    for sup in sorted(out_support.keys()):
        out_support_max[sup] = {}
        for type_gen in out_support[sup]:
            out_support_max[sup][type_gen] = out_support[sup][type_gen][metric].iloc[0]


    for info_i, results in [('time', out_time), (f"max_{metric}", out_support_max), ('FP', out_fp)]:

        info_plot = {}
        for sup in sorted(results.keys()):
            for type_gen in results[sup]:
                type_gen_str = abbreviateValue(f"{type_criterion}_{type_gen}", abbreviations)
                if type_gen_str not in info_plot:
                    info_plot[type_gen_str] = {}
                info_plot[type_gen_str][sup] = results[sup][type_gen]

        figure_name = os.path.join(output_fig_dir, f"{dataset_name}_stree_{min_support_tree}_{metric}_{info_i}.pdf")

        title, ylabel = '', ''
        if info_i == 'time':
            title = 'Execution time'
            ylabel="Execution time $(seconds)$"
        
        elif info_i == f"max_{metric}":
            ylabel="Max $\\Delta_{FPR}$"
            title="Highest $\\Delta_{FPR}$" 

        elif info_i == 'FP':
            ylabel="#FP"
            title="#FP" 

        plotDicts(info_plot, marker=True, \
                title = title, sizeFig=size_fig,\
                        linestyle=lines_style, color_labels=color_labels, \
                xlabel="Minimum support s",ylabel=ylabel  , labelSize=10.2,\
                outside=False,  saveFig=saveFig, nameFig = figure_name)







    # # Store performance results

    if save:
        import os

        output_results = os.path.join(os.path.curdir, output_dir, 'performance')
        from pathlib import Path

        Path(output_results).mkdir(parents=True, exist_ok=True)

        conf_name = f"{dataset_name}_{metric}_{type_criterion}_{min_support_tree}"

        import json
        with open(os.path.join(output_results, f'{conf_name}_time.json'), 'w') as output_file:
            output_file.write(json.dumps(out_time))


        import json
        with open(os.path.join(output_results, f'{conf_name}_fp.json'), 'w') as output_file:
            output_file.write(json.dumps(out_fp))


        out_support_max = {}


        for sup in sorted(out_support.keys()):
            out_support_max[sup] = {}
            for type_gen in out_support[sup]:
                out_support_max[sup][type_gen] = out_support[sup][type_gen][metric].iloc[0]

        with open(os.path.join(output_results, f'{conf_name}_div.json'), 'w') as output_file:
            output_file.write(json.dumps(out_support_max))



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="output_red",
        help="specify the name of the output folder",
    )

    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIRECTORY,
        help="specify the dataset directory",
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

    args = parser.parse_args()

    run_pruning_experiemnt(min_support_tree = args.min_support_tree,
    min_sup_divergences = args.min_sup_divergences,
    type_criterion=args.type_criterion,
    metric = args.metric,
    #save = True,
    #saveFig = True,
        dataset_name = args.dataset_name,
        output_dir=args.name_output_dir,
        dataset_dir=args.dataset_dir,
    )