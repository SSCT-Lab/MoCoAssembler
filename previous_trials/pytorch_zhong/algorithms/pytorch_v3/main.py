import copy
import datetime
import os
import random
import re
import yaml
import utils
from config import marker


def run(model_name, path):
    # Step0: 准备一系列变量
    original_apis, init_api_visited, original_param_kvs, split_dict = {}, {}, {}, {}
    forward_spaces, forward_apis, init_output, forward_output = [], [], [], []

    # Step1: split_model
    file_path_org = os.path.join(marker.SIMPLE_MODEL_PATH, model_name + ".py")
    with open(file_path_org, "r", encoding="utf8") as file_org:
        temp_list = []
        for line in file_org:
            if line.find("super(") >= 0:
                split_dict["super"] = line
                split_dict["part1"] = copy.deepcopy(temp_list)
                temp_list.clear()

            elif line.find("def forward(") >= 0:
                split_dict["forward"] = line
                split_dict["part2"] = copy.deepcopy(temp_list)
                temp_list.clear()

            elif line.find("return x") >= 0:
                split_dict["return"] = line
                split_dict["part3"] = copy.deepcopy(temp_list)
                temp_list.clear()

            else:
                temp_list.append(line)

        split_dict["part4"] = copy.deepcopy(temp_list)
        temp_list.clear()
        file_org.close()

    # pre-process
    for line in split_dict["part2"]:
        if line.strip(' ') == '\n' or line.strip().startswith('#'):
            continue
        lst = line.strip().split("=", 1)

        if "nn." in lst[1]:
            original_apis[lst[0].strip()] = lst[1].strip()
            init_api_visited[lst[0].strip()] = 0
        else:
            init_output.append(line)

    idx = 0
    while split_dict["part2"][idx].strip() == '':
        idx += 1
    init_space = utils.count_space(split_dict["part2"][idx])

    for line in split_dict["part3"]:
        forward_spaces.append(utils.count_space(line))
        if line.strip(' ') == '\n':
            forward_apis.append("\n")
        else:
            forward_apis.append(line.strip())

    # 具体变异逻辑
    for i in range(len(forward_apis)):
        match = re.search(r'self\.\w+(?=\()', forward_apis[i])
        if forward_apis[i].startswith('#') or forward_apis[i] == '\n' or not match:
            forward_sentence = " " * forward_spaces[i] + forward_apis[i]
            if forward_apis[i] != '\n':
                forward_sentence += "\n"
            forward_output.append(forward_sentence)
            continue

        self_name = match.group()
        api_params = original_apis[self_name]
        api_nn = api_params.split('(', 1)[0]
        api_torch = "torch." + api_nn
        api_para_list = api_params.split('(', 1)[1].split(')', -1)[0].split(',')

        new_self_name = self_name + "_" + str(init_api_visited[self_name])
        init_api_visited[self_name] += 1

        init_sentence = " " * init_space + new_self_name + " = "
        forward_sentence = " " * forward_spaces[i] + forward_apis[i].replace(self_name, new_self_name) + "\n"

        try:
            category = utils.get_category(api_torch)
        except Exception:
            init_sentence += api_params + "\n"
            init_output.append(init_sentence)
            forward_output.append(forward_sentence)
            continue

        content = {}
        try:
            constraint_file = os.path.join(marker.CONSTRAINTS_PATH, category, api_torch + ".yaml")
            with open(constraint_file, "r", encoding="utf-8") as f:
                content = yaml.full_load(f.read())
                f.close()
        except Exception as err:
            print(err)

        param_names = content["inputs"]["required"] + content["inputs"]["optional"]
        for idx, para in enumerate(api_para_list):
            if '=' in para:
                kv_pair = para.split('=')
                original_param_kvs[kv_pair[0].strip()] = kv_pair[1].strip()
            else:
                original_param_kvs[param_names[idx]] = para

        if utils.to_mutate_on_api(category):
            try:
                target_api = utils.get_mutate_api(api_torch)
                category = utils.get_category(target_api)
                constraint_file = os.path.join(marker.CONSTRAINTS_PATH, category, target_api + ".yaml")
                with open(constraint_file, "r", encoding="utf-8") as f:
                    content = yaml.full_load(f.read())
                    f.close()
            except Exception:
                init_sentence += api_params + "\n"
                init_output.append(init_sentence)
                forward_output.append(forward_sentence)
                continue
            api_nn = target_api.split('.', 1)[1]

        init_sentence += api_nn + "("

        # 单行变异部分
        constr = content["constraints"]
        input_list = content["inputs"]

        required_list = input_list["required"]
        for idx, item in enumerate(required_list):
            if "range" in constr[item].keys() and constr[item]["range"] == "LIMITED":
                val = original_param_kvs[item]
            else:
                val = utils.mutate_logic(constr[item])

            if category == "linear" and item == "in_features":
                spaces = utils.count_space(forward_sentence)
                forward_output.append(" " * spaces + "x = nn.AdaptiveAvgPool1d(" + str(val) + ")(x)\n")

            init_sentence += item + "=" + str(val) + ", "

        optional_list = input_list["optional"]
        for idx, item in enumerate(optional_list):
            if random.random() < idx / len(optional_list):
                continue

            if item == "return_indices" or item == "groups" or item == "dilation" or item == "padding_idx":
                continue

            val = utils.mutate_logic(constr[item])

            if item == "dim":
                val = -1

            init_sentence += item + "=" + str(val) + ", "

        init_output.append(init_sentence[:-2] + ")\n")
        forward_output.append(forward_sentence)

    # reassemble
    split_dict["part2"], split_dict["part3"] = init_output, forward_output
    file_path_mut = os.path.join(path, model_name + "_mutated.py")
    with open(file_path_mut, "w+", encoding="utf8") as file_mut:
        for line in split_dict["part1"]:
            file_mut.write(line)
        file_mut.write(split_dict["super"])

        for line in split_dict["part2"]:
            file_mut.write(line)

        file_mut.write("\n")
        file_mut.write(split_dict["forward"])

        for line in split_dict["part3"]:
            file_mut.write(line)
        file_mut.write(split_dict["return"])

        for line in split_dict["part4"]:
            file_mut.write(line)
        file_mut.close()


if __name__ == "__main__":
    if not os.path.exists(marker.MUTATED_MODEL_PATH):
        os.makedirs(marker.MUTATED_MODEL_PATH)

    target = os.path.join(marker.MUTATED_MODEL_PATH, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(target)

    model_list = os.listdir(marker.SIMPLE_MODEL_PATH)
    for model in model_list:
        run(model[:-3], target)
