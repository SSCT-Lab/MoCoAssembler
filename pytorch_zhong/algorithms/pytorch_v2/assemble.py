import copy
import datetime
import os

import config.marker as marker
import mutate as mut

# 将整个模型分为四个部分，分别以"super.init()", "def forward()"和"return x"为界
# 第一部分和第四部分不做改变，按照第三部分修改第二部分
split_dict = {}
TARGET_PATH = os.path.join(marker.MUTATED_MODEL_PATH, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


# 整个拆分和组装的过程，将变异的部分独立出来
def assemble():
    file_list = os.listdir(marker.SIMPLE_MODEL_PATH)

    if not os.path.exists(marker.MUTATED_MODEL_PATH):
        os.makedirs(marker.MUTATED_MODEL_PATH)

    os.makedirs(TARGET_PATH)

    for model in file_list:
        model_name = model[:-3]
        split_model(model_name)

        # 将具体的mutate逻辑拆出来
        init_lines, forward_lines = mut.mutate(split_dict["part2"], split_dict["part3"])
        split_dict["part2"], split_dict["part3"] = init_lines, forward_lines

        reassemble_model(model_name)


# 分拆模型，拆成四个列表，方便灵活调整，分离读写过程
def split_model(model_name: str) -> None:
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


# 重新组装，拆分过程的逆过程
def reassemble_model(model_name: str) -> None:
    file_path_mut = os.path.join(TARGET_PATH, model_name + "_mutated.py")
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

    split_dict.clear()


if __name__ == "__main__":
    assemble()
