#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)


def run_compas_experiments_trees(
    name_output_dir="output",
    type_experiment="one_at_time",
    type_criterion="divergence_criterion",
    min_support_tree=0.1,
    min_sup_divergences=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2],
    metric="d_fpr",
    verbose=False,
    ouput_folder_dir=".",
):


    info_list = ["FP", "max"]
    type_gens = ["with_gen", "without_gen"]
    out = {k: {} for k in info_list}
    for i in info_list:
        out[i] = {k: {} for k in type_gens}

    # # Dataset

    dataset_name = "compas"
    risk_class_type = True

    from import_datasets import import_process_compas

    dfI, class_map = import_process_compas(
        risk_class=risk_class_type, continuous_col=True
    )
    dfI.reset_index(drop=True, inplace=True)

    dfI["predicted"] = dfI["predicted"].replace({"Medium-Low": 0, "High": 1})
    true_class_name, pred_class_name = "class", "predicted"
    class_and_pred_names = [true_class_name, pred_class_name]
    attributes = list(dfI.columns.drop(class_and_pred_names))

    dfI = dfI[attributes + class_and_pred_names]

    saveFig = False
    import os

    # # Tree divergence - FPR

    cols_c = ["class", "predicted"]
    continuous_attributes = ["priors_count", "length_of_stay", "age"]
    all_time_results = {}
    
        
    time_results = {"with_gen": {}, "without_gen": {}}


    df_analyze = dfI.copy()

    import time

    start_time_tree = time.time()

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
        type_criterion=type_criterion
    )

    time_results["tree_time"] = time.time() - start_time_tree

    considerOnlyContinuos = True
    if considerOnlyContinuos:
        for k in list(generalization_dict.keys()):
            if k not in continuous_attributes:
                generalization_dict.pop(k, None)


    for min_sup_divergence in min_sup_divergences:
        if verbose:
            print(min_sup_divergence, end=" ")

        # ## Extract divergence - 1 function

        for apply_generalization in [False, True]:
            if apply_generalization == False:
                type_gen = "without_gen"
            else:
                type_gen = "with_gen"
            from utils_extract_divergence_generalized import (
                extract_divergence_generalized,
            )

            allow_overalp = (
                True if type_experiment == "all_attributes" else False
            )
            if (allow_overalp) and (apply_generalization is False):
                continue

            start_time_divergence = time.time()
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
                FPM_type="fpgrowth",
                allow_overalp=allow_overalp,
            )
            time_results[type_gen][min_sup_divergence] = (
                time.time() - start_time_divergence
            )

            from divexplorer_generalized.FP_Divergence import FP_Divergence
            fp_i = FP_Divergence(FP_fm, metric)
            FP_fm = fp_i.getDivergence(th_redundancy=0)

            out["FP"][type_gen][
                float(min_sup_divergence)
            ] = FP_fm.shape[0]
            out["max"][type_gen][
                float(min_sup_divergence)
            ] = max(FP_fm[metric])

    all_time_results = time_results
    if verbose:
        print()

    outputdir = os.path.join(
        ouput_folder_dir,
            name_output_dir,
            dataset_name,
            type_criterion,
            f"stree_{min_support_tree}",
            metric,
    )
    from pathlib import Path

    Path(outputdir).mkdir(parents=True, exist_ok=True)

    import json

    filename = os.path.join(
        outputdir,
        f"info_time.json",
    )
    with open(
        filename,
        "w",
    ) as fp:
        json.dump(all_time_results, fp)

    for i in info_list:
        output = out[i]

        filename = os.path.join(
            outputdir,
            f"info_ALL_{i}.json",
        )

        with open(
            filename,
            "w",
        ) as fp:
            json.dump(output, fp)



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_output_dir",
        default="output_res",
        help="specify the name of the output folder",
    )

    parser.add_argument(
        "--type_criterion",
        type=str,
        default="divergence_criterion",
        help='specify the experiment type among ["divergence_criterion", "entropy"]',
    )

    parser.add_argument(
        "--min_sup_tree",
        type=float,
        default=0.1,
        help="specify the minimum support for the tree induction",
    )

    parser.add_argument(
        "--show_fig",
        action="store_true",
        help="specify show_fig to show the tree graph.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="specify verbose to print working in progress status.",
    )

    parser.add_argument(
        "--no_compute_divergence",
        action="store_true",
        help="specify no_compute_divergence to not compute the divergence scores.",
    )

    parser.add_argument(
        "--type_experiments",
        nargs="*",
        type=str,
        default=["one_at_time", "all_attributes"],
        help="specify the types of experiments to evaluate among ['one_at_time', 'all_attributes']",
    )
    parser.add_argument(
        "--min_sup_divs",
        nargs="*",
        type=float,
        default=[
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
        ],
        help="specify a list of min supports of interest, with values from 0 to 1",
    )

    parser.add_argument(
        "--metrics",
        nargs="*",
        type=str,
        default=["d_fpr"],  # , "d_fnr", "d_error"]
        help="specify a list of metric of interest, ['d_fpr', 'd_fnr', 'd_error', 'd_accuracy]",
    )

    args = parser.parse_args()

    run_compas_experiments_trees(
        type_criterion=args.type_criterion,
        name_output_dir=args.name_output_dir,
        type_experiments=args.type_experiments,
        min_support_tree=args.min_sup_tree,
        min_sup_divergences=args.min_sup_divs,
        show_fig=args.show_fig,
        metrics=args.metrics,
        verbose=args.verbose,
        no_compute_divergence=args.no_compute_divergence,
    )
