import os


def depart(model_name):

    # 获取当前文件路径
    dir_cur = os.getcwd()
    dir_org = dir_cur + "/model/tensorflow_version/"
    dir_tmp = dir_cur + "/template/tensorflow_version/"

    input_tensor = "input_tensor"
    output_tensor = "output_tensor"

    # 如果模版文件不存在，则创建新文件
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)

    file_path_org = dir_org + model_name + ".py"
    file_path_tmp = dir_tmp + model_name + "_tmp.py"
    file_path_res = dir_tmp + model_name + "_res.py"

    file_tmp = open(file_path_tmp, "w+", encoding="utf8")
    file_res = open(file_path_res, "w+", encoding="utf8")

    with open(file_path_org, "r", encoding="utf8") as file_org:

        for line in file_org:
            if line.find(input_tensor) >= 0:
                file_tmp.write("# " + model_name + " input layer" + "\n")
                file_tmp.write(line)
                break
            file_tmp.write(line)

        for line in file_org:
            if line.find(output_tensor) >= 0:
                break
            file_res.write(line)

        file_tmp.write("\n")
        file_tmp.write("# " + model_name + " output layer" + "\n")

        for line in file_org:
            file_tmp.write(line)

        file_org.close()
        file_tmp.close()


if __name__ == "__main__":
    path = os.getcwd()
    model_dir = path + "/model/tensorflow_version/"
    file_list = os.listdir(model_dir)
    net_list = []
    
    # 仅在python文件内操作
    for i in file_list:
        if i.endswith(".py"):
            net_list.append(i[:-3])

    for net in net_list:
        depart(net)
