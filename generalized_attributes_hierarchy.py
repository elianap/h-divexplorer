def flatten_list(list_i):
    import itertools

    flatten = itertools.chain.from_iterable
    return list(flatten([i for i in list_i]))


def is_inner_a_frozenset(list_i):
    return any(isinstance(el, frozenset) for el in list_i)


def is_Frozenset(el):
    return type(el) is frozenset


class generalized_attribute_detail:
    def __init__(
        self, attribute, value, generalized_value, indexAttr, indexAttr_flatten
    ):
        self.attribute = attribute
        self.attribute_value = value
        self.generalized_value = generalized_value
        self.indexAttr = indexAttr

        self.indexAttr_list = (
            frozenset([indexAttr]) if type(indexAttr) != frozenset else indexAttr
        )
        # self.indexAttr_lista = (
        #     [indexAttr] if type(indexAttr) != frozenset else list(indexAttr)
        # )
        self.indexAttr_flatten = indexAttr_flatten
        self.name = f"{attribute}={generalized_value}"

    def __str__(self):
        return f"{self.attribute}={self.attribute_value} ==> {self.generalized_value}  (indexAttr: {self.indexAttr}) (indexAttr_list: {self.indexAttr_list}) flatten: {self.indexAttr_flatten}"


class Generalizations_hierarchy:
    def __init__(self, counter_id, attribute_value_incompatible={}):
        # self.map_generalizations = {}
        self.attribute_ids_gen = {}
        self.ids_gen = {}
        self.generalizations_id = []
        self.counter_id = counter_id
        self.map_generalization_objs = {}
        self.attribute_value_incompatible = attribute_value_incompatible

    # def list_id_generalizations(self):
    #     self.generalizations_id = []
    #     for gens in self.map_generalizations.values():
    #         for list_gens in gens.values():
    #             self.generalizations_id.append(
    #                 frozenset([g.indexAttr for g in list_gens])
    #             )
    #     return self.generalizations_id

    def list_id_generalizations(self):
        generalizations_id_obj = []
        for gens in self.map_generalization_objs.values():
            for general_obj in gens.values():
                generalizations_id_obj.append(general_obj.get_ids())

        return generalizations_id_obj

    def list_id_generalizations_flatten(self):
        generalizations_id_obj = []
        for gens in self.map_generalization_objs.values():
            for general_obj in gens.values():
                generalizations_id_obj.append(general_obj.get_ids_flatten())

        return generalizations_id_obj

    def dict_id_generalizations_flatten(self):
        generalizations_id_obj = {}
        for gens in self.map_generalization_objs.values():
            for general_obj in gens.values():
                generalizations_id_obj[
                    general_obj.id_gen_attr
                ] = general_obj.get_ids_flatten()

        return generalizations_id_obj

    def dict_name_id_generalizations_flatten(self):
        generalizations_id_obj = {}
        for gens in self.map_generalization_objs.values():
            for general_obj in gens.values():
                generalizations_id_obj[
                    (general_obj.id_gen_attr, general_obj.name)
                ] = general_obj.get_ids_flatten()

        return generalizations_id_obj

    # def get_generalizations(self):
    #     return self.map_generalizations

    def extend_dataset_with_hierarchy(self, df_one_hot, drop_intervals=None, sep="=" ):
        from utils_hierarchy import extend_dataset_with_hierarchy

        return extend_dataset_with_hierarchy(df_one_hot, self, drop_intervals=drop_intervals, sep=sep)

    def add_generalizations(
        self, dict_generalization, attributes_one_hot, level_struct=True, verbose=False
    ):

        if level_struct:
            from utils_hierarchy import convertGeneralizationInLevels

            dict_generalization_level_sorted = convertGeneralizationInLevels(
                dict_generalization
            )

            for attr, level_gens in dict_generalization_level_sorted.items():
                # Add level by level: input dict are sorted
                for gens in level_gens.values():
                    for v, gen_v in gens.items():
                        if f"{attr}={v}" in attributes_one_hot:
                            index_attr = attributes_one_hot.index(f"{attr}={v}")
                            index_attr_flatten = frozenset([index_attr])
                        else:
                            if self.check_presence(attr, v, verbose=verbose):
                                index_attr = self.get_id_generalized(attr, v)
                                index_attr_flatten = self.get_id_generalized_flatten(
                                    attr, v
                                )
                            else:
                                continue

                        if type(gen_v) == list:
                            for gen_v_i in gen_v:
                                self.add_generalization(
                                    attr, v, gen_v_i, index_attr, index_attr_flatten
                                )
                        else:
                            self.add_generalization(
                                attr, v, gen_v, index_attr, index_attr_flatten
                            )
        # TODO
        else:
            for attr, gens in dict_generalization.items():
                for v, gen_v in gens.items():
                    if f"{attr}={v}" in attributes_one_hot:
                        index_attr = attributes_one_hot.index(f"{attr}={v}")
                        index_attr_flatten = frozenset([index_attr])
                    else:
                        if self.check_presence(attr, v, verbose=verbose):
                            index_attr = self.get_id_generalized(attr, v)
                            index_attr_flatten = self.get_id_generalized_flatten(
                                attr, v
                            )
                        else:
                            continue
                    self.add_generalization(
                        attr, v, gen_v, index_attr, index_attr_flatten
                    )

    def check_presence(self, attribute, generalized_value, verbose=False):
        return self.check_presence_attribute(
            attribute, verbose=verbose
        ) and self.check_presence_generalized_value(
            attribute, generalized_value, verbose=verbose
        )

    def check_presence_attribute(self, attribute, verbose=False):
        if attribute not in self.map_generalization_objs:
            if verbose:
                print(f"Attribute {attribute} not available")
            return False
        return True

    def check_presence_generalized_value(
        self, attribute, generalized_value, verbose=False
    ):
        if generalized_value not in self.map_generalization_objs[attribute]:
            if verbose:
                print(f"Value {attribute} - {generalized_value} not available")
            return False
        return True

    def get_id_generalized(self, attribute, generalized_value):
        return self.map_generalization_objs[attribute][generalized_value].get_ids()

    def get_id_generalized_flatten(self, attribute, generalized_value):
        return self.map_generalization_objs[attribute][
            generalized_value
        ].get_ids_flatten()

    def add_generalization(
        self, attribute, value, generalized_value, indexAttr, indexAttr_flatten
    ):
        if attribute not in self.map_generalization_objs:
            self.map_generalization_objs[attribute] = {}
        if generalized_value not in self.map_generalization_objs[attribute]:
            self.map_generalization_objs[attribute][
                generalized_value
            ] = GeneralizedAttribute(attribute, generalized_value, self.counter_id)
            self.counter_id = self.counter_id + 1
        self.map_generalization_objs[attribute][
            generalized_value
        ].add_generalization_attribute(value, indexAttr, indexAttr_flatten)

    def name_mapping_flatten(self):
        name_mapping_flatten = {}
        for attribute, gen_d in self.map_generalization_objs.items():
            for generalized_value, generalized_obj in gen_d.items():
                name_mapping_flatten[
                    frozenset(generalized_obj.get_ids_flatten())
                ] = f"{attribute}={generalized_value}"
        return name_mapping_flatten

    def get_attributes_incompatibility(self):
        incompatible_attribute = self.attribute_value_incompatible

        for attribute, gen_d in self.map_generalization_objs.items():
            for generalized_obj in gen_d.values():
                if attribute not in incompatible_attribute:
                    incompatible_attribute[attribute] = frozenset([])
                incompatible_attribute[attribute] = incompatible_attribute[
                    attribute
                ].union(generalized_obj.get_incompatible())
        return incompatible_attribute

    def get_attributes_incompatibility_remapped(self, remapped):
        incompatible_attribute = self.attribute_value_incompatible

        for attribute, gen_d in self.map_generalization_objs.items():
            for generalized_obj in gen_d.values():
                if attribute not in incompatible_attribute:
                    incompatible_attribute[attribute] = frozenset([])
                incompatible_attribute[attribute] = incompatible_attribute[
                    attribute
                ].union(generalized_obj.get_incompatible())
        return incompatible_attribute

    def __str__(self):
        str_ret = ""
        for attribute, gen_d in self.map_generalization_objs.items():
            attribute_name = f"Attribute: {attribute}\n"
            str_ret_lev_1 = ""
            for gen_value, generalized_obj in gen_d.items():
                s = f"{gen_value}\nInfo:{ generalized_obj.__str__()}"
                str_ret_lev_1 = f"{str_ret_lev_1}\n{s}\n"
            str_ret = f"{str_ret}{attribute_name}{str_ret_lev_1}"
        return str_ret

    # def __str__(self):
    #     str_ret = ""
    #     for attribute, gen_d in self.map_generalization_objs.items():
    #         attr_name = f"{attribute}\n"
    #         str_ret_lev_1 = ""
    #         for gen_value, generalized_obj in gen_d.items():
    #             s = gen_value
    #             s = f"{s}\n{generalized_obj.__str__()}"
    #             str_ret = f"{str_ret}{s}"

    #         str_ret = f"{str_ret}{attr_name}{str_ret_lev_1}"
    #     return str_ret


class GeneralizedAttribute:
    def __init__(self, attribute, generalized_value, new_id):
        self.attribute = attribute  # redundant
        self.generalized_value = generalized_value
        self.id_gen_attr = new_id
        self.list_generalizations = []
        # self.incompatible = frozenset([new_id])
        self.name = f"{attribute}={generalized_value}"

    def add_generalization_attribute(self, value, indexAttr, indexAttr_flatten):
        self.list_generalizations.append(
            generalized_attribute_detail(
                self.attribute,
                value,
                self.generalized_value,
                indexAttr,
                indexAttr_flatten,
            )
        )

    def get_ids(self):
        return frozenset([g.indexAttr for g in self.list_generalizations])

    def get_ids_flatten(self):
        f_list = frozenset([])
        for g in self.list_generalizations:
            f_list = f_list.union(g.indexAttr_flatten)
        return frozenset(f_list)

    def get_incompatible(self):
        return frozenset(self.get_ids_flatten()).union([self.id_gen_attr])

    # def __str__(self):

    #     t = ", ".join(
    #         [f"{g.attribute}={g.attribute_value}" for g in self.list_generalizations]
    #     )
    #     return f"{self.id_gen_attr} {self.attribute}={self.generalized_value} {self.get_ids()} {t} {self.get_incompatible()}"

    def __str__(self):

        t = ", ".join(
            [f"{g.attribute}={g.attribute_value}" for g in self.list_generalizations]
        )
        return f"{self.attribute}={self.generalized_value} (id:{self.id_gen_attr})  {t}   (IDS gen: {self.get_ids_flatten()}) Incompatibles{self.get_incompatible()}"


class Itemset_Generalized:
    def __init__(self, itemset_fr, name, support):
        self.itemset_fr = itemset_fr
        self.name = name
        self.support = support

    def __str__(self):
        return f"{self.itemset_fr} {self.name} {self.support}"
