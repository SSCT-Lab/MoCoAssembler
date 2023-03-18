import os


def depart(model_name):
    # 得到当前目录
    dir_cur = os.getcwd()

    # 得到源模型目录和修改过模型的保存目录
    dir_org = dir_cur + "/model/pytorch_version/simple_models/"
    dir_tmp = dir_cur + "/template/pytorch_version/"

    # 若修改过模型的目标目录暂时不存在则创建
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)

    # 得到修改前全名和修改后全名（绝对路径）
    file_path_org = dir_org + model_name + ".py"
    file_path_tmp = dir_tmp + model_name + "_template.py"

    file_tmp = open(file_path_tmp, "w+", encoding="utf8")

    with open(file_path_org, "r", encoding="utf8") as file_org:
        for line in file_org:
            if line.find("super(") >= 0:
                file_tmp.write(line)
                break
            file_tmp.write(line)
        file_tmp.write("\n")

        for line in file_org:
            if line.find("def forward(") >= 0:
                file_tmp.write(line)
                break
        file_tmp.write("\n")
        for line in file_org:
            if line.find("return x") >= 0:
                file_tmp.write(line)
                break
        file_tmp.write("\n")

        for line in file_org:
            file_tmp.write(line)

        file_org.close()
        file_tmp.close()


if __name__ == "__main__":
    path = os.getcwd()
    model_dir = path + "/model/pytorch_version/simple_models/"
    file_list = os.listdir(model_dir)
    net_list = []
    for i in file_list:
        if i.endswith(".py"):
            net_list.append(i[:-3])

    for net in net_list:
        depart(net)
