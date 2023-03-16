import json
import re


def delete_code_and_get_api():
    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    # with open(file, "r", encoding="utf-8") as f1,open("delete_code.txt", "a", encoding="utf-8") as f2:
    #     flag=False#标记要改写的代码
    #     for line in f1:
    #         if "KitModel" in line:
    #             flag=True
    #         if "def" in line and "KitModel" not in line:
    #             flag=False
    #         if flag:
    #             if re.search(r'    \w.*=', line) != None :
    #                 f2.write(line)

    # 取到api
    result = {}  # empty dict
    with open("delete_code.txt", "r", encoding="utf-8") as f1:
        for line in f1:
            left, right = line.split('=', 1)
            key = re.findall(r"    .*_\d", left)
            print(key)
            key = "".join(key)
            result.setdefault(key, []).append(line)
        json_data = json.dumps(result)

    with open("apis.json", "w", encoding="utf-8") as f1:
        f1.write(json_data)


def add_API_to_model(file):
    code = []
    with open(file, "r", encoding="utf-8") as f1:
        for line in f1:
            code.append(line)
            if re.search("weights_dict = load_weights_from_file", line) is not None:
                with open("apis.json", 'r+') as f2:
                    content = f2.read()
                    content = json.loads(content)
                    # if content[]
                    input_layers = content.get(list(content.keys())[0])
                    for i in range(len(input_layers)):
                        code.append(input_layers[i])
                    output_layers = content.get(list(content.keys())[-2])
                    for i in range(len(output_layers)):
                        code.append(output_layers[i])
                    last = content.get(list(content.keys())[-1])
                    for i in range(len(last)):
                        code.append(last[i])
    for i in range(len(code)):
        with open("new_model.py", "a", encoding="utf-8") as f1:
            if "set_layer_weights(model, weights_dict)" in code[i]:
                continue
            f1.write(code[i])


def remove_API_from_model(file):
    code = []
    import random
    with open(file, "r", encoding="utf-8") as f1, open("apis.json", 'r+') as f2:
        content = f2.read()
        content = json.loads(content)
        length = len(list(content.keys()))
        print("长度是：", length)
        # 不要去掉第一层,不要去掉最后一层
        num = random.randint(2, length - 1)
        print("抽到的数字是", num)
        # 要去掉的层是
        rm_layer = list(content.keys())[num]
        print("rm_layer:", rm_layer)
        # 去掉的层之前的层是
        pre_layer = list(content.keys())[num - 1]
        print("prelayer", pre_layer)
        # 记录之前层的最后一个变量名，以修改代码
        pre_layer_last_name = content.get(pre_layer)
        print("pre_layer_last_name", pre_layer_last_name)
        future_name = pre_layer_last_name[-1]
        future_name = "".join(future_name)
        future_name_left, future_name_right = future_name.split("=", 1)
        future_name_left = future_name_left.lstrip()
        print(future_name_left)
        rmed_layer_name = content.get(rm_layer)[-1]
        print(rmed_layer_name)
        rm_left_name, rm_right_name = rmed_layer_name.split("=", 1)
        rm_left_name = "".join(rm_left_name)
        rm_left_name = rm_left_name.lstrip()
        rm_left_name = rm_left_name.strip()
        print("rm_left_name", rm_left_name)
        if "flatten" in rm_layer:
            pass
        else:
            for line in f1:
                if line in content.get(rm_layer):
                    continue
                else:
                    code.append(line)
    cishu = str(num)
    filename = "new_model_" + cishu + ".py"
    with open(filename, "a", encoding="utf-8") as f1:
        for i in range(len(code)):
            if "    set_layer_weights" in code[i]:
                continue
            # 更改后续代码
            if rm_left_name in code[i]:
                str1 = code[i]
                str1 = str1.replace(rm_left_name, future_name_left)
                f1.write(str1)
            else:
                f1.write(code[i])


if __name__ == '__main__':
    # delete_code_and_get_api("alexnet_keras.py")
    # add_API_to_model("orignal.py")
    remove_API_from_model("alexnet_keras.py")

    # with open("./alexnet_model/converted.json","r",encoding="utf-8") as f1:
    #     new_data=json.load(f1)
    #     print(type(new_data))
