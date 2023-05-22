import os
import random
import yaml
from typing import List

import marker
import algorithms.pytorch_v1.similarity_crud as simi


# 变异init函数中的API名称，找同类API进行替换
def mutate_on_api(line: str) -> str:
    api = "torch." + line.split('=', 1)[1].strip()
    related_api_dict = simi.read_similarity(api)

    mutated_line = line

    if len(related_api_dict) == 0:
        print("API not included.")
    else:
        # 对line进行处理
        print(related_api_dict)

    # return mutated_line
    return "#mutated1\n\n"


# 变异init函数中API的参数，例如kernel_size等
def mutate_on_params(line: str) -> str:
    # 获取需要变异的api
    complete_api = "torch." + line.split('=', 1)[1].strip()
    api_name = complete_api.split('(')[0].strip()
    file = os.path.join(marker.PT_ROOT_PATH, "constraints", "category_mapping.yaml")
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        content = yaml.full_load(content)
        f.close()

    mutated_line = line

    if api_name not in content.keys():
        print("API not included.")
    else:
        category = content[api_name]
        file = os.path.join(marker.PT_ROOT_PATH, "constraints", "pytorch_modified", category, api_name + ".yaml")
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            content = yaml.full_load(content)
            f.close()
        # 对content进行处理
        print(content)

    # return mutated_line
    return "#mutated2\n\n"


# 确认forward函数删改部分
def delete_on_forward(lines: List[str], modified_line: str) -> str:
    regex = modified_line.split('=')[0].strip()

    result = ""
    for line in lines:
        result += line
        if regex in line:
            break

    return result


api_num_dict = {"alexnet_ptver": 12, "bilstm_ptver": 6, "densenet_ptver": 20, "gru_ptver": 6, "lenet_ptver": 11,
                "lstm_ptver": 16, "mobilenet_ptver": 30, "squeezenet_ptver": 22, "vgg16_ptver": 23, "vgg19_ptver": 26}


# 是否进行变异，这里采用等概率选择行，易算得选择的概率为 1/cur_num
def to_mutate(cur_num: int) -> bool:
    return random.random() < 1 / cur_num


# 选择变异api还是params
def to_mutate_on_api() -> bool:
    return random.randint(0, 1) == 1


def mutate(file_path: str, file_name: str):
    # 创建变异后模型保存路径
    mutated_path = os.path.join(marker.PT_ROOT_PATH, "mutated_models")
    if not os.path.exists(mutated_path):
        os.makedirs(mutated_path)

    original_file = os.path.join(file_path, file_name + ".py")
    mutated_file = os.path.join(mutated_path, file_name + "_mutated.py")

    # 为等概率选到目标api，需要知道当前模型的api总数
    api_num = api_num_dict[file_name]

    file_mutated = open(mutated_file, "w+", encoding="utf8")
    with open(original_file, "r", encoding="utf8") as f:
        for line in f:
            if line.find("super(") >= 0:
                file_mutated.write(line)
                break
            file_mutated.write(line)

        init_mutate_end = False
        modified_line = ""

        # init 部分变异
        for line in f:
            if line.find("def forward(") >= 0:
                file_mutated.write(line)
                break

            # 在init部分pytorch的API调用都以nn.开头
            if line.find("nn.") >= 0 and not init_mutate_end:
                if to_mutate(api_num):
                    modified_line = line
                    if to_mutate_on_api():
                        line = mutate_on_api(line)
                    else:
                        line = mutate_on_params(line)
                    init_mutate_end = (line != modified_line)
                api_num -= 1
            file_mutated.write(line)

        # forward部分删减（确定最后一行）
        ops_list = []
        temp = ""
        for line in f:
            if line.find("return x") >= 0:
                temp = line
                break
            ops_list.append(line)

        content_mutated = delete_on_forward(ops_list, modified_line)
        file_mutated.write(content_mutated)

        file_mutated.write(temp)
        for line in f:
            file_mutated.write(line)

        f.close()
        file_mutated.close()


if __name__ == '__main__':
    model_dir = os.path.join(marker.PT_ROOT_PATH, "original_models", "simple_models")
    file_list = os.listdir(model_dir)

    for net in file_list:
        mutate(model_dir, net[:-3])
