import yaml

from config.paths import tf_mut_file, tf_tmp_file, tf_func_file, tf_param_file
from config.keywords import INPUT_TENSOR, INF
from pathlib import Path
import random
import re
import json


def mutate(model_name):

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
    mutate_list = [mutate_on_parma,
                   mutate_on_function,]

    for line in file_res:
        """
        空格或者备注就写入变异文件中
        网络结构就进行变异
        """
        if line == "\n" or line.find("#") >= 0:
            file_mut.write(line)

        else:
            """
            变异选择算法
            """

            if is_mutate_on_parma():
                file_mut.write(mutate_on_parma(line))
            elif is_mutate_on_function():
                file_mut.write(mutate_on_function(line))


def mutate_select() -> list:

    pass


def is_mutate_on_parma() -> bool:
    return True


def is_mutate_on_function() -> bool:
    return False


def mutate_on_parma(line: str) -> str:
    """
    :param line: 一行api
    :return: 新生成的api
    """
    param_file = Path.joinpath(tf_param_file, "new")

    function, dict = depart_function(line)

    func_file = "tf." + function + ".yaml"
    func_file = param_file / func_file
    if func_file.exists():
        with Path.open(func_file, "r") as file:
            data = yaml.load(file, yaml.Loader)
            data = data["constraints"]
            params_dict = dict.copy()
            """
            随机选择一个参数进行变异"""
            param = random.choice(list(data.keys()))
            value = ""
            if "dtype" in data[param]:
                dtype = data[param]["dtype"]
                type = random.choice(dtype) if isinstance(dtype, list) else dtype

                match type:
                    case "tf.string":
                        if "enum" in data[param]:
                            value = random.choice((data[param]["enum"]))
                            value = '"' + value + '"'
                        else:
                            value = data[param]["default"]
                    case "tf.bool":
                        value = random.choice([True, False])
                    case "float":
                        value = random.random().__str__()
                    case "int":
                        if "range" in data[param]:
                            """
                            所有的range范围均为[0,inf)
                            """
                            value = random.randint(0, INF).__str__()
                        elif "structure" in data[param]:
                            structure = random.choice(data[param]["structure"])
                            """
                            此处的value值随便取的，应该做一定的更改
                            """
                            match structure:
                                case "int":
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


def depart_function(line: str) -> (str, dict):
    """
    :param line: 一行api
    :return: (函数名， 参数列表)
    """
    function = re.findall(r".*?x = (.*?)\(.*?", line)[0]

    infos_1 = re.findall(r".*?\((?P<param>.*?)\)\((?P<input>.*?)\)", line)

    params = infos_1[0][0] + ','
    input = infos_1[0][1]

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
    :param line: 一行api
    :param params_dict: 新生成的参数列表
    :return: 新生成的api
    """
    new_params = ""
    for _ in params_dict:
        new_params = new_params + _ + "=" + params_dict[_].__str__() + ", "

    new_params = new_params[:-2]
    new_params = "(" + new_params + ")"

    org_params = re.findall(r".*?(\(.*?\)).*?", line, re.S)[0]

    new_line = line.replace(org_params, new_params, 1)
    return new_line


def mutate_on_function(line: str) -> str:
    """
    :param line: 一行api
    :return: 新生成的api
    """
    function = re.findall(".*?x = (.*?)\(.*?", line)[0]
    func_dir = tf_func_file
    # 返回一个迭代器
    func_list = Path.rglob(func_dir, "*.json")

    for _ in func_list:
        if _.__str__().find(function) >= 0:
            with Path.open(Path.joinpath(func_dir, Path("/"), _), "r") as func_mut:
                mut_list = json.load(func_mut)
                mut_func = random.choice(mut_list)[0]
                """可能需要更改mut_func的参数以及输入类型"""
                line = line.replace(function, mut_func)
            break

    return line


if __name__ == "__main__":
    mutate("a_test")
