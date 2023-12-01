import os
import random

from _database import database
from Concrete import concrete
from Mutate import get_mutator
from ResultAnalyse import analyser
from BoundaryGenerator import generateBoundary
import os.path as p
import yaml
import copy


config = {"N": 3, "LIBRARY_LIST": []}
generateBoundary = generateBoundary


def set_config():
    CONFIG_PATH = p.join("..", "config", "config.yaml")
    f = open(CONFIG_PATH, "r", encoding="utf-8")
    file_config = yaml.full_load(f)
    f.close()
    config["LIBRARY_LIST"].clear()
    for library in list(file_config["LIBRARY_LIST"].values()):
        config["LIBRARY_LIST"].append(library)
    config["N"] = file_config["N"]


def refresh__config():
    set_config()
    concrete.new_experiment()
    database.refresh_config()
    concrete.refresh_config()
    analyser.refresh_config()


def check() -> bool:
    MAIN_PATH = ".."
    REPORT_PATH = p.join(MAIN_PATH, "report")
    RESULT_PATH = p.join(MAIN_PATH, "result")
    if not p.exists(REPORT_PATH):
        os.makedirs(REPORT_PATH)
    if not p.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    return True


def find_half_with_min_value(dictionary):
    # 按值排序字典项，并获取前一半的键
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1])
    half_length = len(sorted_items) // 2
    half_with_min_value = [item[0] for item in sorted_items[:half_length]]
    return half_with_min_value


def goFuzzing(net: str = "LeNet") -> None:
    check()
    set_config()
    mutator = get_mutator(config["LIBRARY_LIST"][0])
    seed = database.get_seed(net)
    concrete.set_model_name(net)
    report_f = open(p.join(concrete.get_experiment_path(), "main_report.txt"), "w", encoding="utf-8")
    report_f.write("id: " + concrete.get_experiment_id() + "\npath: " + concrete.get_experiment_path() + "\n\n")
    report_f.close()
    report_f = open(p.join(concrete.get_experiment_path(), "main_report.txt"), "a", encoding="utf-8")
    count_pass = 0
    count_err = 0
    count = 0
    template: dict = {net: []}
    child_model_name_list = list(seed.keys())[1:]
    child_models = {}  # child model name -> child model structure list
    for child_model_name in child_model_name_list:
        child_models[child_model_name] = seed[child_model_name]
    for model in seed.keys():
        if model != net:
            template[model] = seed[model]
    queue = [template]
    layer = 0
    for ele in seed[net]:
        this_layer_count = 0
        this_layer_pass = 0
        this_layer_err = 0
        abandoned_case_num = 0
        if len(queue) == 0:
            break
        if (not isinstance(ele, dict)) or ("layer" in ele.keys() and (ele["layer"] == "cat" or ele["layer"] == "add")):
            for model in queue:
                model[net].append(ele)
                # just for cat and add now
        elif isinstance(ele, dict) and "layer" in ele.keys():
            pass_model_list = []
            pass_gen_index_list = []
            if database.is_abstract_api_name_valid(ele["layer"]):
                for model in queue:
                    for i in range(config["N"]):
                        temp = copy.deepcopy(model)
                        mutated_dict, mutate_info = mutator.mutate(ele)
                        temp[net].append(mutated_dict)
                        gen = layer + 1
                        index = this_layer_count + 1
                        this_layer_count += 1
                        result = concrete.perform(temp, gen, index)
                        for r in result:
                            r["mutate info"] = mutate_info
                            if gen <= 3:
                                r["train test"] = True
                                r["train time cost"] = 1.0
                        ana = analyser.analyse_result(result)
                        if ana:
                            pass_model_list.append(temp)
                            pass_gen_index_list.append((gen, index))
                            count_pass += 1
                            this_layer_pass += 1
                            print(net + "-" + str(gen) + "-" + str(index) + " 组装完成, 成功")
                        else:
                            count_err += 1
                            this_layer_err += 1
                            print(net + "-" + str(gen) + "-" + str(index) + " 组装完成, 失败")
            elif ele["layer"] in child_model_name_list:
                child_model_to_mutate = {ele["layer"]: child_models[ele["layer"]]}
                for model in queue:
                    for i in range(config["N"]):
                        temp = copy.deepcopy(model)
                        mutated_dict, mutate_info = mutator.mutate(child_model_to_mutate)
                        new_layer = copy.deepcopy(ele)
                        new_child_model_name = list(mutated_dict.keys())[0]
                        new_layer["layer"] = new_child_model_name
                        temp[net].append(new_layer)
                        temp[new_child_model_name] = mutated_dict[new_child_model_name]
                        gen = layer + 1
                        index = this_layer_count + 1
                        this_layer_count += 1
                        result = concrete.perform(temp, gen, index)
                        for r in result:
                            r["mutate info"] = mutate_info
                        ana = analyser.analyse_result(result)
                        if ana:
                            pass_model_list.append(temp)
                            pass_gen_index_list.append((gen, index))
                            count_pass += 1
                            this_layer_pass += 1
                            print(net + "-" + str(gen) + "-" + str(index) + " 组装完成, 成功")
                        else:
                            count_err += 1
                            this_layer_err += 1
                            print(net + "-" + str(gen) + "-" + str(index) + " 组装完成, 失败")

            # BRANCH CUTTING
            print("scanning and cutting branch...")
            direction_list = os.listdir(concrete.get_experiment_path())
            gen = layer + 1
            tag = "-" + str(gen) + "-"
            cutting_dict: dict[str: dict[tuple: float]] = {}
            for direction_name in direction_list:
                if tag in direction_name:
                    index = int(direction_name.split("-")[-1])
                    if (gen, index) not in pass_gen_index_list:
                        continue
                    else:
                        with open(p.join(concrete.get_experiment_path(), net + tag + str(index), "report.txt"), "r") as f:
                            info = f.read()
                        train_time_cost = float(info.split("train time cost: ")[1].split("\n", 1)[0])
                        mutate_info = info.split("mutate info: ")[1]
                        if mutate_info in cutting_dict.keys():
                            cutting_dict[mutate_info][(gen, index)] = train_time_cost
                        else:
                            cutting_dict[mutate_info]: dict[tuple: float] = {(gen, index): train_time_cost}
            save_list = []
            for key in cutting_dict:
                now_dict = cutting_dict[key]
                if len(now_dict) == 1:
                    current_save_list = now_dict.keys()
                else:
                    current_save_list = find_half_with_min_value(now_dict)
                for tu in current_save_list:
                    save_list.append(tu)
            abandoned_case_num = len(pass_gen_index_list) - len(save_list)
            save_model_list = []
            for gen_index in save_list:
                save_model_list.append(pass_model_list[pass_gen_index_list.index(gen_index)])
            pass_model_list = save_model_list
            # BRANCH CUTTING

            # generate MoCo_NA file
            library_name = config["LIBRARY_LIST"][0]
            # if you want to generate for cases cut too, change this to "pass_gen_index_list".
            # if you don't want to generate for cases cut, change this to "save_list".
            for gen_index in pass_gen_index_list:
                gen, index = gen_index[0], gen_index[1]
                target_path = p.join(concrete.get_experiment_path(), net + "-" + str(gen) + "-" + str(index))
                # get shape
                with open(p.join(target_path, "report.txt"), "r", encoding="utf-8") as ff:
                    info = ff.read()
                    shape_str = info.split("shape: ", 1)[1].split("\n")[0]
                    shape_num = len(shape_str.split(","))
                    # shape_ints = shape_str.replace("[", "").replace("]", "").split(",")
                shape_max = [8, 8, 224, 224, 224]
                shape_str = ""
                for i in range(shape_num):
                    shape_str += str(random.randint(1, shape_max[i]))
                    shape_str += ", "
                if len(shape_str) <= 2:
                    shape_str = "1, 3, 224, 224"
                else:
                    shape_str = shape_str[:-2]
                if library_name in ["torch", "jittor"]:
                    input_sentence = "import " + library_name + "\n\nx = " + library_name + \
                                     ".randn(" + shape_str + ")\n"
                else:
                    input_sentence = "import " + library_name + " as tf\n\nx = tf.random.normal(shape=[" + shape_str + "])\n"
                # get final definition
                file_list = os.listdir(target_path)
                py_file = ""
                for file_name in file_list:
                    if file_name.endswith(".py"):
                        py_file = file_name
                if library_name in ["torch", "jittor"]:
                    with open(p.join(target_path, py_file), "r", encoding="utf-8") as ff:
                        info = ff.read()
                        desperate_flag = "    def forward" if library_name == "torch" else "    def execute"
                        target_line = info.split(desperate_flag, 1)[0].split("\n")[-3].split(" = ", 1)[1]
                        input_sentence += "layer = " + target_line + "\n"
                        input_sentence += "y = layer(x)\n"
                else:
                    # TODO tensorflow
                    with open(p.join(target_path, py_file), "r", encoding="utf-8") as ff:
                        info = ff.read()
                        desperate_flag = "\n    # output layers"
                        target_line = info.split(desperate_flag, 1)[0].split("\n")[-2].replace("    ", "") + "\n"
                        input_sentence += target_line
                # generate py file
                with open(p.join(target_path, "MoCoNA.py"), "w", encoding="utf-8") as ff:
                    ff.write(input_sentence)
            # generate MoCo_NA file

            queue = pass_model_list
            layer += 1
        report = str(layer) + "层: 成功" + str(this_layer_pass) + "个, 失败" + str(this_layer_err) + "个, 通过剪枝丢弃掉" + str(abandoned_case_num) + "个。 \n"
        report_f.write(report)
        print(report)
    report = "总: 成功" + str(count_pass) + "个, 失败" + str(count_err) + "个。\n"
    report_f.write(report)
    print(report)
    report_f.close()


# if __name__ == "__main__":
#     go("LeNet")
