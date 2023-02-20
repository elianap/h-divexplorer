import pandas as pd

OUTCOME = "outcome"
CLASSIFICATION = "classification"

from utils_extract_divergence_generalized_ranking import check_target_inputs, define_target

def discretize_df_via_discretizations(
    dfI: pd.DataFrame,
    discretizations: dict,
    allow_overalp:bool=False
):

    from copy import deepcopy


    # ## Discretize the dataset

    # Discretize the dataset using the obtained discretization ranges

    from utils_discretization import discretizeDataset_from_relations

    df_s_discretized, discretized_attr = discretizeDataset_from_relations(
        dfI, discretizations, ret_original_attrs=False, allow_overalp=allow_overalp
    )
    if allow_overalp:
        return df_s_discretized
    else:
        return df_s_discretized[dfI.columns]
