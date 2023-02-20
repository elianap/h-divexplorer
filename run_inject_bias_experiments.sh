for experiment in one_at_time all_attributes 
do
    echo "Experiment: $experiment"
    for criterion in entropy weighted_sum_abs_reference_s
    do 
        echo "Criterion: $criterion"
        for gain in 0.01 0.005 0.001 0.0005 0.0025 
        do
            echo "Gain: $gain"
            for i in 0.1 0.15 0.125 0.075 0.05  #0.005 0.025 0.01 
            do
                echo "Min sup tree: $i"
                python experiments_inject_bias.py --noise --no_diverg_compute --type_criterion $criterion --minimal_gain $gain --min_sup_tree $i --name_output_dir result_artificial_injected_bias_40_noise_float --type_experiment $experiment

            done
        done
    done
done
