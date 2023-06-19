import copy
import os

import file_paths

# 重构完成, 已经可以使用新方法了

# split the model given into four parts, and store them into a dictionary
# part1: head(import, class, def_init), part2: layers declaration, part3: layers execute, part 4: test. type: list[str]
# horizon parts: super, execute, return. type: str
def split_simple_model(model_path: str):
    split_dictionary = {}
    file_path_org = model_path
    with open(file_path_org, "r", encoding="utf8") as file_org:
        temp_list = []
        for line in file_org:
            if line.find("super(") >= 0:
                split_dictionary["super"] = line
                split_dictionary["part1"] = copy.deepcopy(temp_list)
                temp_list.clear()

            elif line.find("def execute(") >= 0:
                split_dictionary["execute"] = line
                split_dictionary["part2"] = copy.deepcopy(temp_list)
                temp_list.clear()

            elif line.find("return x") >= 0:
                split_dictionary["return"] = line
                split_dictionary["part3"] = copy.deepcopy(temp_list)
                temp_list.clear()

            else:
                temp_list.append(line)

        split_dictionary["part4"] = copy.deepcopy(temp_list)
        temp_list.clear()
        file_org.close()
        return split_dictionary


def split_model(model_path: str) -> dict:
    return split_simple_model(model_path)


if __name__ == '__main__':
    dic = split_simple_model(os.path.join(file_paths.SIMPLE_MODEL_PATH, 'lenet.py'))
