import os


def depart(model_name):
    # 得到当前目录
    dir_cur = os.getcwd()

    # 得到源模型目录和修改过模型的保存目录
    dir_org = dir_cur + "/model/tensorflow_version/"
    dir_tmp = dir_cur + "/template/tensorflow_version/"

    # 定义 input_tensor/ output_tensor
    input_tensor = "input_tensor"
    output_tensor = "output_tensor"

    # 若修改过模型的目标目录暂时不存在则创建
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)

    # 得到输入参数模型的原名字和修改后名字
    model_name_org = model_name
    model_name_tmp = model_name + "_template"

    # 得到修改前全名和修改后全名（绝对路径）
    file_path_org = dir_org + model_name_org + ".py"
    file_path_tmp = dir_tmp + model_name_tmp + ".py"

    file_tmp = open(file_path_tmp, "w+", encoding="utf8")

    with open(file_path_org, "r", encoding="utf8") as file_org:

        # 在源文件中遍历每行，找input_tensor
        for line in file_org:
            if line.find(input_tensor) >= 0:
                file_tmp.writelines("# " + model_name + " input layer" + "\n")
                file_tmp.write(line)
                break
            file_tmp.write(line)

        file_tmp.write("\n")

        for line in file_org:
            if line.find(output_tensor) >= 0:
                file_tmp.write("# " + model_name + " output layer" + "\n")
                file_tmp.write(line)
                break

        for line in file_org:
            file_tmp.write(line)

        file_org.close()
        file_tmp.close()


if __name__ == "__main__":
    path = os.getcwd()
    model_dir = path + "/model/tensorflow_version/"
    # 列出绝对路径下的文件名列表
    file_list = os.listdir(model_dir)
    # 只针对python文件进行操作
    net_list = []
    for i in file_list:
        if i.endswith(".py"):
            net_list.append(i[:-3])

    for net in net_list:
        depart(net)

    # net_list = ["alexnet_tfver",
    #             "densenet_tfver",
    #             "googlenet_tfver",
    #             "inceptionv3_tfver",
    #             "lenet_tfver",
    #             "mobilenet_tfver",
    #             "resnet18_tfver",
    #             "resnet50_tfver",
    #             "squeezenet_tfver",
    #             "vgg16_tfver",
    #             "vgg19_tfver",
    #             "xception_tfver"]
