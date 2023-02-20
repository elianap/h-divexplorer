from TreeDivergence_unique import TreeDivergence_unique

OUTCOME = "outcome"
CLASSIFICATION = "classification"

import warnings

warnings.filterwarnings("ignore")


class TreeDiscretization_unique:
    def __init__(self):
        self.trees = None
        self.generalization_dict = None
        self.discretizations = None
        self.type_splitting = None
        self.preserve_interval = None

    def check_is_fitted(self):
        if not hasattr(self, "fitted"):
            raise ValueError(
                "This instance is not fitted yet. Call 'get_tree_discretization' with appropriate arguments before using this tree discretizer"
            )

    def printDiscretizationTrees(self, round_v=5):
        self.check_is_fitted()
        if self.trees is None:
            import warnings

            warnings.warn("Tree not available/stored")
            return
        if type(self.trees) is dict:
            for attribute, tree_div in self.trees.items():
                print(attribute)
                tree_div.printTree(round_v=round_v)
                print()
        else:
            print(f"Type splitting: {self.type_splitting}")
            self.trees.printTree(round_v=round_v)

    def get_tree_discretization(
        self,
        dfI,
        min_support=0.1,
        metric="d_fpr",
        class_map={"N": 0, "P": 1},
        attributes=None,
        continuous_attributes=None,
        class_and_pred_names=None,  # ["class", "predicted"],
        target_col=None,
        storeTree=False,
        verbose=False,
        **kwargs,
    ):

        if target_col is None and class_and_pred_names is None:
            raise ValueError(
                f"Please, specify the target: the class and predicted columns names in class_and_pred_names for classification purposes or target_col in case of single target (e.g. rankings)."
            )

        if target_col is not None and class_and_pred_names is not None:
            raise ValueError(
                f"Please, specify only one target: the class and predicted columns names in class_and_pred_names for classification purposes or target_col in case of single target (e.g. rankings)."
            )
        if class_and_pred_names is not None:
            target_type = CLASSIFICATION
        else:
            target_type = OUTCOME

        if attributes is None:
            if target_type == CLASSIFICATION:
                attributes = list(dfI.columns.drop(class_and_pred_names))
            if target_type == OUTCOME:
                attributes = list(dfI.columns.drop(target_col))

        if continuous_attributes is None:
            continuous_attributes = attributes
            raise Warning(
                "Continuous attributes are not specified. Assume all in attributes are continuous"
            )


        generalization_dict, discretizations = self.one_at_the_time_discretization(
                dfI,
                continuous_attributes,
                min_support=min_support,
                metric=metric,
                class_map=class_map,
                class_and_pred_names=class_and_pred_names,
                target_col=target_col,  # Added target col
                target_type=target_type,  # Added target col
                storeTree=storeTree,
                verbose=verbose,
                continuous_attributes=continuous_attributes,
                **kwargs,
        )

        # For now we keep it as a separate step
        # TODO: integrate in the discretization range generation process
        # Note: this step is "useless" if we are splitting only continuos attribute by default
        considerOnlyContinuos = True
        if considerOnlyContinuos:
            for k in list(discretizations.keys()):
                if k not in continuous_attributes:
                    discretizations.pop(k, None)

        self.generalization_dict = generalization_dict
        self.discretizations = discretizations
        self.fitted = True

        return generalization_dict, discretizations


    def one_at_the_time_discretization(
        self,
        dfI,
        input_attributes,
        min_support=0.1,
        metric="d_fpr",
        class_map={"N": 0, "P": 1},
        class_and_pred_names=["class", "predicted"],
        target_col=None,  # Added target col
        target_type=CLASSIFICATION,  # Added target col
        storeTree=False,
        verbose=False,
        continuous_attributes=None,
        **kwargs,
    ):

        if target_type == CLASSIFICATION:
            target_columns = class_and_pred_names
            configs_target = {
                "target_type": target_type,
                "class_name": class_and_pred_names[0],
                "pred_name": class_and_pred_names[1],
                "target_name": None,
            }
        else:
            target_columns = [target_col]
            configs_target = {
                "target_type": target_type,
                "class_name": None,
                "pred_name": None,
                "target_name": target_col,
            }

        generalization_dict = {}
        discretizations = {}
        if storeTree and input_attributes != []:
            self.trees = {}
        for attribute_s in input_attributes:
            df_s = dfI.copy()
            df_s = df_s[[attribute_s] + target_columns]

            tree_div = TreeDivergence_unique(**configs_target)

            tree = tree_div.generateTree(
                df_s,
                class_map,
                metric=metric,
                minimum_support=min_support,
                continuous_attributes=continuous_attributes,
                **kwargs,
            )

            (
                generalization_dict_i,
                discretization_i,
            ) = tree_div.getHierarchyAndDiscretizationSplits(tree, verbose=verbose)
            if generalization_dict_i is None and discretization_i is None:
                import warnings

                msg = f"Attribute {attribute_s} could not be discretized: you should lower the support threshold (<{min_support})."
                # warnings.warn(msg)
            else:
                generalization_dict.update(generalization_dict_i)
                discretizations.update(discretization_i)
                if storeTree:
                    self.trees[attribute_s] = tree_div
        return generalization_dict, discretizations

    def get_number_nodes(self):
        if self.trees is None:
            # import warnings
            # warnings.warn("Tree discretization is none")
            return 0, 0
        if type(self.trees) is dict:
            n_internal_nodes, n_leaf_nodes = 0, 0
            for attribute in self.trees:
                n_internal_nodes += self.trees[attribute].get_number_internal_nodes()
                n_leaf_nodes += self.trees[attribute].get_number_leaf_nodes()
        else:
            n_internal_nodes, n_leaf_nodes = 0, 0
            n_internal_nodes = self.trees.get_number_internal_nodes()
            n_leaf_nodes = self.trees.get_number_leaf_nodes()
        return n_internal_nodes, n_leaf_nodes


    def get_keep_items_associated_with_divergence(self):

        """
        Given the original generalization_dict and discretization_dict, it keeps only the items that are indeed associated with a divergent behavior.
        Hence, the ones that positevely contribute to the divergence.

        Args:
            self
            generalization_dict (dict): the original dictionary of generalizatiion
            discretizations (dict): the original dictionary of discretization

        Returns
            dict : new generalization, with only positively divergent terms
            dict: new discretization, with only positively divergent terms
        """
        items_keep = {}
        for attribute in self.trees:
            items_keep[attribute] = self.trees[
                attribute
            ].get_keep_items_for_attributes()
        return items_keep
