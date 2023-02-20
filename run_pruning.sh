
for criterion in divergence_criterion
do 
    echo "Criterion: $criterion"
    for dataset_name in  adult german compas artificial_gaussian #online_shoppers_intention 
    do
        echo "dataset_name: $dataset_name"

        python run_pruned_v2.py --take_top_k 1000 --metric d_fpr --dataset_name $dataset_name --type_criterion $criterion  --min_sup_divergences 0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 --name_output_dir output_results_12_28

    done
done
