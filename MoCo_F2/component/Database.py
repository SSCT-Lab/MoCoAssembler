import copy
import os
import os.path as p
import yaml
import csv

database_path = p.join("..", "database")
implicit_database_path = p.join(database_path, "implicit")
config_path = p.join("..", "config", "config.yaml")


def merge(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    intersection_list = list(intersection)
    return intersection_list


class Database:
    def __init__(self):
        self.__library_list = []
        self.__threshold = 0.0
        self.__mode = 0
        self.__read_config()
        self.__implicit_layer_info = {}
        self.__implicit_layer_similarity = {}
        self.__implicit_layer_similarity_valid = {}
        self.__api_name_map = {}
        self.__api_name_map_valid = {}
        self.__inverse_api_name_map = {}
        self.__inverse_api_name_map_valid = {}
        self.__all_api_list = {}
        self.__candidate_map = {}
        self.__api_para_map = {}
        self.__abstract_api_layer_info = {}
        self.__total_refresh()

    def __total_refresh(self):
        self.__read_implicit_layer_info()
        self.__read_implicit_layer_similarity()
        self.__read_api_name_map_and_inverse_api_name_map()
        self.__calculate_api_name_map_valid()
        self.__modify_api_name_map_valid()
        self.__calculate_inverse_api_name_map_valid()
        self.__calculate_all_api_list()
        self.__calculate_implicit_layer_similarity_valid()
        self.modify_similarity_dict_with_shape()
        self.__calculate_candidate_map()
        self.__read_api_para_map()
        self.__calculate_abstract_api_layer_info()
        print("### Database OK.")

    def __part_refresh_for_threshold(self):
        self.__calculate_implicit_layer_similarity_valid()
        self.__calculate_candidate_map()
        self.modify_similarity_dict_with_shape()
        print("### Database OK.")

    def __read_config(self):
        # unit test passed
        print("### initializing config")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.full_load(f)
        # check
        assert (config["MODE"] == 1 and len(config["LIBRARY_LIST"].values()) == 1) or \
               (config["MODE"] == 2 and len(config["LIBRARY_LIST"].values()) >= 2), "MODE and LIST are not matched"
        library_dict = config["LIBRARY_LIST"]
        self.__library_list = list(library_dict.values())
        self.__threshold = config["THRESHOLD"]
        self.__mode = config["MODE"]
        return

    def __read_implicit_layer_info(self):
        # unit test passed
        print("### initializing layer info")
        self.__implicit_layer_info = {}
        dir_list = os.listdir(implicit_database_path)
        for library in self.__library_list:
            assert library in dir_list, "check config"
        for library in self.__library_list:
            current_library_layer_info = {}
            current_layer_info_path = p.join(implicit_database_path, library, "layer_info")
            file_list = os.listdir(current_layer_info_path)
            for file_name in file_list:
                current_file_path = p.join(current_layer_info_path, file_name)
                with open(current_file_path, "r", encoding="utf-8") as f:
                    current_info = yaml.full_load(f)
                    current_layer_name = file_name[:-5]
                    current_library_layer_info[current_layer_name] = current_info
            self.__implicit_layer_info[library] = current_library_layer_info
        return

    def __read_implicit_layer_similarity(self):
        # unit test passed
        print("### initializing similarity")
        self.__implicit_layer_similarity = {}
        for library in self.__library_list:
            current_similarity_file = p.join(implicit_database_path, library, "layer_similarity.yaml")
            with open(current_similarity_file, "r", encoding="utf-8") as f:
                self.__implicit_layer_similarity[library] = yaml.full_load(f)
        return

    def __read_api_name_map_and_inverse_api_name_map(self):
        # unit test passed
        print("### initializing maps")
        self.__api_name_map = {}
        self.__inverse_api_name_map = {}
        target_csv_path = p.join(database_path, "abstract", "abstract_api_name.csv")
        f = open(target_csv_path, "r")
        reader = csv.reader(f)
        row_list = []
        for row in reader:
            if len(row) > 2:
                row_list.append(row)
        head = row_list[0]
        row_list = row_list[1:]
        valid_locations = []
        for library in self.__library_list:
            valid_locations.append(head.index(library))
        self.__api_name_map = {}
        self.__inverse_api_name_map = {}
        for library in self.__library_list:
            self.__inverse_api_name_map[library] = {}
        for row in row_list:
            api_id = row[0]
            abstract_api_name = row[1]
            implicit_library_api_name = []
            for location in valid_locations:
                implicit_library_api_name.append(row[location])
            dict_for_main_map = {"id": api_id}
            for i in range(len(self.__library_list)):
                dict_for_main_map[self.__library_list[i]] = implicit_library_api_name[i] if \
                    len(implicit_library_api_name[i]) > 5 else "NOAPI"
            self.__api_name_map[abstract_api_name] = dict_for_main_map
            for i in range(len(self.__library_list)):
                self.__inverse_api_name_map[self.__library_list[i]][implicit_library_api_name[i]] = abstract_api_name
            for i in range(len(self.__library_list)):
                if "" in self.__inverse_api_name_map[self.__library_list[i]].keys():
                    self.__inverse_api_name_map[self.__library_list[i]].pop("")
                if "NOAPI" in self.__inverse_api_name_map[self.__library_list[i]].keys():
                    self.__inverse_api_name_map[self.__library_list[i]].pop("NOAPI")
        return

    def __calculate_api_name_map_valid(self):
        # unit test passed
        print("### calculating valid api map")
        self.__api_name_map_valid.clear()
        for abstract_api_name in self.__api_name_map:
            implicit_apis = self.__api_name_map[abstract_api_name].values()
            if "" not in implicit_apis and "NOAPI" not in implicit_apis:
                self.__api_name_map_valid[abstract_api_name] = self.__api_name_map[abstract_api_name]
        return

    def __modify_api_name_map_valid(self):
        if self.__mode == 2:
            return
        valid_implicit_api_name_list = []
        file_list = os.listdir(p.join(implicit_database_path, self.__library_list[0], "layer_info"))
        for file_name in file_list:
            valid_implicit_api_name_list.append(file_name[:-5])
        old_dict = copy.deepcopy(self.__api_name_map_valid)
        for key in old_dict:
            if old_dict[key][self.__library_list[0]] not in valid_implicit_api_name_list:
                self.__api_name_map_valid.pop(key)
        return

    def __calculate_inverse_api_name_map_valid(self):
        # unit test passed
        print("### calculating valid inverse map")
        self.__inverse_api_name_map_valid.clear()
        valid_list = list(self.__api_name_map_valid.keys())
        for library in self.__library_list:
            old_dict = self.__inverse_api_name_map[library]
            new_dict = {}
            for implicit_api_name in old_dict.keys():
                if old_dict[implicit_api_name] not in valid_list:
                    continue
                else:
                    new_dict[implicit_api_name] = old_dict[implicit_api_name]
            self.__inverse_api_name_map_valid[library] = new_dict
        return

    def __calculate_all_api_list(self):
        # unit test passed
        print("### calculating api range")
        self.__all_api_list = {"abstract": list(self.__api_name_map_valid.keys())}
        for library in self.__library_list:
            self.__all_api_list[library] = self.__inverse_api_name_map_valid[library].keys()
        return

    def __calculate_implicit_layer_similarity_valid(self):
        # unit test passed
        print("### calculating valid similarity")
        self.__implicit_layer_similarity_valid = {}
        for library in self.__library_list:
            old_dict = self.__implicit_layer_similarity[library]
            new_dict = {}
            for api in old_dict.keys():
                if not self.is_implicit_api_name_valid(library, api):
                    continue
                current_dict = old_dict[api]
                new_current_dict = {}
                for simi_api in current_dict:
                    if simi_api == api or not self.is_implicit_api_name_valid(library, simi_api):
                        continue
                    elif current_dict[simi_api] < self.__threshold:
                        continue
                    else:
                        new_current_dict[simi_api] = current_dict[simi_api]
                new_dict[api] = new_current_dict
            self.__implicit_layer_similarity_valid[library] = new_dict
        return

    def __calculate_candidate_map(self):
        print("### calculating candidate map")
        self.__candidate_map = self.__just_get_candidate_dict(self.__threshold)
        pass

    def __read_api_para_map(self):
        # TODO
        self.__api_para_map.clear()
        if self.__mode == 1:
            abstract_para_name_path = p.join(database_path, "abstract", "abstract_para_name_new.yaml")
            f = open(abstract_para_name_path, "r", encoding="utf-8")
            data = yaml.full_load(f)
            f.close()
            self.__api_para_map = data
            # for abstract_api_name in self.__all_api_list["abstract"]:
            #     self.__api_para_map[abstract_api_name] = {}
            #     implicit_api_name = self.get_implicit_api_name(self.__library_list[0], abstract_api_name)
            #     implicit_layer_info = self.get_implicit_layer_info(self.__library_list[0], implicit_api_name)
            #     paras = list(implicit_layer_info["constraints"].keys())
            #     for para in paras:
            #         abstract_para_name = para
            #         self.__api_para_map[abstract_api_name][abstract_para_name] = {}
            #         self.__api_para_map[abstract_api_name][abstract_para_name][self.__library_list[0]] = para
        else:
            pass

    def __calculate_abstract_api_layer_info(self):
        # TODO
        self.__abstract_api_layer_info.clear()
        if self.__mode == 1:
            for abstract_api_name in self.__all_api_list["abstract"]:
                self.__abstract_api_layer_info[abstract_api_name] = self.get_implicit_layer_info(
                    self.__library_list[0], self.get_implicit_api_name(self.__library_list[0], abstract_api_name))
        else:
            pass

    # validation check functions
    def is_library_valid(self, library: str) -> bool:
        return library in self.__library_list

    def is_abstract_api_name_valid(self, abstract_api_name: str) -> bool:
        return abstract_api_name in self.__all_api_list["abstract"]

    def is_implicit_api_name_valid(self, library: str, implicit_api_name: str) -> bool:
        return self.is_library_valid(library) and implicit_api_name in self.__all_api_list[library]

    # information functions
    def get_implicit_api_similarity_valid(self, library: str, implicit_api_name: str) -> dict:
        assert self.is_library_valid(library) and self.is_implicit_api_name_valid(library, implicit_api_name),\
            "check input"
        return self.__implicit_layer_similarity_valid[library][implicit_api_name]

    def get_implicit_layer_info(self, library: str, implicit_api_name: str) -> dict:
        assert self.is_library_valid(library) and self.is_implicit_api_name_valid(library, implicit_api_name),\
            "check input"
        return self.__implicit_layer_info[library][implicit_api_name]

    def get_implicit_api_name(self, library: str, abstract_api_name: str) -> str:
        assert self.is_library_valid(library), "check input library"
        assert self.is_abstract_api_name_valid(abstract_api_name), "check input abstract_api_name"
        return self.__api_name_map_valid[abstract_api_name][library]

    def get_abstract_api_name(self, library: str, implicit_api_name: str) -> str:
        assert self.is_library_valid(library), "check input library"
        assert self.is_implicit_api_name_valid(library, implicit_api_name), "check input"
        return self.__inverse_api_name_map_valid[library][implicit_api_name]

    def get_abstract_layer_info(self, abstract_api_name: str) -> dict:
        assert self.is_abstract_api_name_valid(abstract_api_name), "check input"
        # TODO
        if self.__mode == 1:
            return self.__abstract_api_layer_info[abstract_api_name]
        else:
            return {}

    def get_implicit_para_name(self, library: str, abstract_api_name: str, abstract_para_name: str) -> str:
        assert self.is_library_valid(library) and self.is_abstract_api_name_valid(abstract_api_name), "check input"
        # TODO
        if self.__mode == 1:
            return self.__api_para_map[abstract_api_name][abstract_para_name][library]
        else:
            return " "

    def get_seed(self, seed_name: str) -> dict:
        if self.__library_list[0] == "tensorflow":
            SEED_PATH = p.join(database_path, "seed2", seed_name + ".yaml")
        else:
            SEED_PATH = p.join(database_path, "seed", seed_name + ".yaml")
        f = open(SEED_PATH, "r")
        seed = yaml.full_load(f)
        f.close()
        return seed

    # config functions
    def set_threshold(self, threshold: float):
        assert 0.01 <= threshold <= 0.99, "check threshold, between 0.01 and 0.99."
        self.__threshold = threshold
        self.__candidate_map = self.__just_get_candidate_dict(self.__threshold)
        print("Threshold changed to " + str(threshold) + ", map updated.")
        return

    def get_threshold(self):
        return self.__threshold

    def refresh_config(self):
        old_mode = copy.deepcopy(self.__mode)
        old_list = copy.deepcopy(self.__library_list)
        old_threshold = copy.deepcopy(self.__threshold)
        self.__read_config()
        if old_mode != self.__mode or old_list != self.__library_list:
            self.__total_refresh()
        elif old_threshold != self.__threshold:
            self.__part_refresh_for_threshold()
        return

    # about candidate list calculation
    def get_candidate_mutate_list(self, abstract_api_name: str) -> list[str]:
        assert self.is_abstract_api_name_valid(abstract_api_name)
        return self.__candidate_map[abstract_api_name]

    def __get_candidate_mutate_list(self, abstract_api_name: str, threshold: float) -> list[str]:
        abstract_candidate_api_list = self.__all_api_list["abstract"]
        for library in self.__library_list:
            current_implicit_api_name = self.get_implicit_api_name(library, abstract_api_name)
            if current_implicit_api_name == "NOAPI" or current_implicit_api_name == "":
                continue
            current_abstract_candidate_api_list = []
            implicit_candidate_api_list = \
                list(self.__implicit_layer_similarity_valid[library][current_implicit_api_name].keys())
            for layer in implicit_candidate_api_list:
                current_abstract_candidate_api_list.append(self.get_abstract_api_name(library, layer))
            abstract_candidate_api_list = merge(abstract_candidate_api_list, current_abstract_candidate_api_list)
        if abstract_api_name in abstract_candidate_api_list:
            abstract_candidate_api_list.remove(abstract_api_name)
        return abstract_candidate_api_list

    def __get_candidate_dict(self, threshold: float) -> (dict, int):
        zero_num = 0
        # abstract_api_num = len(self.main_api_name_map)
        result_dict = {}
        abstract_api_list = self.__all_api_list["abstract"]
        for abstract_api_name in abstract_api_list:
            candidate_list = self.__get_candidate_mutate_list(abstract_api_name, threshold)
            if len(candidate_list) == 0:
                zero_num += 1
            result_dict[abstract_api_name] = candidate_list
        return result_dict, zero_num

    def __just_get_candidate_dict(self, threshold: float) -> dict:
        return self.__get_candidate_dict(threshold)[0]

    def __generate_candidate_report(self, threshold: float) -> (str, int):
        candidate_dict, zero_num = self.__get_candidate_dict(threshold)
        report = "*With the threshold of " + str(threshold) + ", there are " + str(zero_num) + \
                 " apis that cannot be mutated\n\n"
        for abstract_api_name in candidate_dict.keys():
            report += (abstract_api_name + " :\n")
            report += str(candidate_dict[abstract_api_name])
            report += "\n"
        return report, zero_num

    def generate_candidate_report(self) -> None:
        report = self.__generate_candidate_report(threshold=self.__threshold)[0]
        report_path = p.join("..", "report", "apiCandidateReport.txt")
        f = open(report_path, "w", encoding="utf-8")
        f.write(report)
        f.close()
        return

    # about candidate list calculation

    def modify_similarity_dict_with_shape(self) -> None:
        if self.__mode != 1:
            return
        similarity_valid_dict = copy.deepcopy(self.__implicit_layer_similarity_valid[self.__library_list[0]])
        for implicit_api_name in similarity_valid_dict.keys():
            if '1d' in implicit_api_name:
                pool = ['2d', '3d']
            elif '2d' in implicit_api_name:
                pool = ['1d', '3d']
            elif '3d' in implicit_api_name:
                pool = ['1d', '2d']
            else:
                continue
            for simi_implicit_api_name in similarity_valid_dict[implicit_api_name].keys():
                if pool[0] in simi_implicit_api_name or pool[1] in simi_implicit_api_name:
                    self.__implicit_layer_similarity_valid[self.__library_list[0]][implicit_api_name].pop(simi_implicit_api_name)
                else:
                    continue
        return


# d = Database()
