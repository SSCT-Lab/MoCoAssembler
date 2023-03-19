import os
import random
from typing import List

import pytorch_api as api


# 变异init函数中的API名称，找同类API进行替换
def mutate_on_api(line: str) -> str:
    return api.change_api(line)


# 变异init函数中API的参数，例如kernel_size等
def mutate_on_params(line: str) -> str:
    return api.change_params(line)


# 变异forward函数，例如增删指令，变换次序等
def mutate_on_forward(lines: List[str]) -> str:
    return "# forward\n"


# 是否进行变异，这边暂时用随机数表示变异与否，后期优化该选择算法
def to_mutate() -> bool:
    return random.randint(0, 1) == 1


def mutate(file_path: str, file_name: str):
    mutated_path = os.getcwd() + "/pytorch_models/"
    if not os.path.exists(mutated_path):
        os.makedirs(mutated_path)

    original_file = file_path + file_name + ".py"
    mutated_file = mutated_path + file_name + "_mutated.py"

    file_mutated = open(mutated_file, "w+", encoding="utf8")

    with open(original_file, "r", encoding="utf8") as f:
        for line in f:
            if line.find("super(") >= 0:
                file_mutated.write(line)
                break
            file_mutated.write(line)

        # init 部分变异
        for line in f:
            if line.find("def forward(") >= 0:
                file_mutated.write(line)
                break

            # 在init部分pytorch的API调用都以nn.开头
            if line.find("nn.") >= 0:
                if to_mutate():
                    line = mutate_on_api(line)
                if to_mutate():
                    line = mutate_on_params(line)

            file_mutated.write(line)

        # forward部分变异
        ops_list = []
        temp = ""
        for line in f:
            if line.find("return x") >= 0:
                temp = line
                break
            ops_list.append(line)

        content_mutated = mutate_on_forward(ops_list)
        file_mutated.write(content_mutated)

        file_mutated.write(temp)
        for line in f:
            file_mutated.write(line)

        f.close()
        file_mutated.close()
    return


if __name__ == '__main__':
    path = os.getcwd()
    model_dir = path + "/../model/pytorch_version/simple_models/"
    file_list = os.listdir(model_dir)

    for net in file_list:
        mutate(model_dir, net[:-3])
