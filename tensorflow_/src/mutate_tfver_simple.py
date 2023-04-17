import yaml

from config.paths import tf_mut_file, tf_tmp_file, tf_param_file, tf_func_sim_file
from config.keywords import INPUT_TENSOR, INF
from pathlib import Path
import random
import re


def mutate(model_name):
    """
    变异函数主体
    :param model_name: 模型名
    """

    # 获取模版文件
    dir_tmp = tf_tmp_file
    dir_mut = tf_mut_file

    # 如果变异文件不存在，则新建
    if not Path.exists(dir_mut):
        Path.mkdir(dir_mut)

    input_tensor = INPUT_TENSOR

    file_path_tmp = Path.joinpath(dir_tmp, (model_name + "_tmp.py"))
    file_path_res = Path.joinpath(dir_tmp, (model_name + "_res.py"))
    file_path_mut = Path.joinpath(dir_mut, (model_name + "_mut.py"))

    file_tmp = Path.open(file_path_tmp, "r", encoding="utf8")
    file_res = Path.open(file_path_res, "r", encoding="utf8")

    with Path.open(file_path_mut, "w+", encoding="utf8") as file_mut:

        # 写入输入层
        for line in file_tmp:
            file_mut.write(line)
            if line.find(input_tensor) >= 0:
                break

        # 对于网络结构进行变异
        mutate_on_model(file_res, file_mut)

        # 写入输出层
        for line in file_tmp:
            file_mut.write(line)

        file_res.close()
        file_tmp.close()


def mutate_on_model(file_res, file_mut):
    """
    对模型进行变异
    :param file_res: 函数所在的文件
    :param file_mut: 变异后的模型文件
    :return:
    """
    # 两种变异方法
    mutate_list = [mutate_on_parma,
                   mutate_on_function]

    for line in file_res:
        if line == "\n" or line.find("#") >= 0:
            file_mut.write(line)

        else:
            # 随机选择一种变异方法进行变异
            method = random.choice(mutate_list)
            if method == mutate_list[0]:
                file_mut.write("# 参数列表变异\n")
            else:
                file_mut.write("# 函数变异\n")
            file_mut.write(method(line))


def mutate_on_parma(line: str) -> str:
    """
    :param line: 一行api
    :return: 新生成的api
    """
    param_file = tf_param_file
    function, dict = depart_line(line)

    func_file = "tf." + function + ".yaml"
    func_file = param_file / func_file
    if func_file.exists():
        with Path.open(func_file, "r") as file:
            data = yaml.load(file, yaml.Loader)
            data = data["constraints"]
            params_dict = dict.copy()
            # 随机选择一个参数进行变异
            param = random.choice(list(data.keys()))
            value = ""
            if "dtype" in data[param]:
                dtype = data[param]["dtype"]
                type = random.choice(dtype) if isinstance(dtype, list) else dtype

                # 参数类型匹配
                match type:
                    case "tf.string":
                        if "enum" in data[param]:
                            value = random.choice((data[param]["enum"]))
                            value = '"' + value + '"'
                        else:
                            value = data[param]["default"]
                            if isinstance(value, str):
                                value = '"' + value + '"'
                    case "tf.bool":
                        value = random.choice([True, False])
                    case "float":
                        value = random.random().__str__()
                    case "int":
                        if "range" in data[param]:
                            # 可能需要改范围
                            value = random.randint(0, INF).__str__()
                        elif "structure" in data[param]:
                            structure = random.choice(data[param]["structure"])
                            # 此处的value值随便取的，应该做一定的更改
                            match structure:
                                case "integer":
                                    value = random.randint(1, 4)
                                case "tuple":
                                    value = tuple(random.randint(1, 4) for _ in range(data[param]["shape"]))
                                case "list":
                                    value = list(random.randint(1, 4) for _ in range(data[param]["shape"]))
                                case "tuple_of_tuples":
                                    value = tuple(tuple(random.randint(1, 4) for _ in range(2)) for _ in
                                                  range(data[param]["shape"]))
                        else:
                            value = data[param]["default"]
            else:
                value = data[param]["default"]
            params_dict[param] = value
            new_line = generate_param_line(line, params_dict)
    else:
        new_line = line
    return new_line


def depart_line(line: str) -> (str, dict):
    """
    :param line: 一行api
    :return: (函数名， 参数列表)
    """
    function = re.findall(r".*?x = (.*?)\(.*?", line)[0]

    infos_1 = re.findall(r".*?\((?P<param>.*?)\)\((?P<input>.*?)\)", line)

    params = infos_1[0][0] + ','

    infos_2 = re.findall(r".*?\((.*?)\).*?", params, re.S)
    for _ in infos_2:
        __ = _.replace(" ", "")
        params = params.replace(_, __)

    params_dic = {}
    details = re.finditer(r"(?P<param>.*?)=(?P<value>.*?),", params, re.S)
    for _ in details:
        params_dic.update({_["param"].strip(): _["value"].strip()})

    return function, params_dic


def generate_param_line(line, params_dict) -> str:
    """
    根据参数列表生成新的api
    :param line: 一行api
    :param params_dict: 新生成的参数列表
    :return: 新生成的api
    """
    new_params: str = ""
    for _ in params_dict:
        new_params = new_params + _ + "=" + params_dict[_].__str__() + ", "

    new_params = new_params[:-2]
    new_params = "(" + new_params + ")"

    org_params: str = re.findall(r".*?(\(.*?\)).*?", line, re.S)[0]

    new_line = line.replace(org_params, new_params, 1)
    return new_line


def mutate_on_function(line: str) -> str:
    """
    :param line: 一行api
    :return: 新生成的api
    """
    function, dict = depart_line(line)
    func_file = "tf." + function + ".yaml"
    func_file = tf_func_sim_file / func_file

    # 相似度阈值
    th = 0.6
    if func_file.exists():
        with Path.open(func_file, "r") as file:
            data = yaml.load(file, yaml.Loader)
            # 有的函数大于阈值的相似度可能只有它本身，但是变异的时候又不想要他本身，所以简单的处理一下数据
            lst = [_[0] for _ in data.items() if _[1] > th]
            func_mut = random.choice(lst[1:])
            param_file = tf_param_file / (func_mut + ".yaml")
            tmp_dict = {}
            if param_file.exists():
                with Path.open(tf_param_file / (func_mut + ".yaml"), "r") as param_file:
                    params_dict = yaml.load(param_file, yaml.Loader)["constraints"]
                    for _ in dict.items():
                        if _[0] in params_dict.keys():
                            tmp_dict[_[0]] = _[1]
            else:
                # 有的函数的参数列表取值可能没有储存，所以直接copy不改变其参数列表（可能会出错）
                tmp_dict = dict.copy()

            line = line.replace(function, func_mut)
            new_line = generate_param_line(line, tmp_dict)

    return new_line


if __name__ == "__main__":
    mutate("a_test")
