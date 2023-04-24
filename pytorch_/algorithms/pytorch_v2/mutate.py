import random
import re
from typing import List

import descp_loader as loader
import similarity as simi
import category_mapper as mapper

original_apis, init_api_visited, original_param_kvs = {}, {}, {}
forward_spaces, forward_apis = [], []
init_output, forward_output = [], []

INF_VAL, LOOP_TIMES = 4, 10


def mutate(init_line_list: List[str], forward_line_list: List[str]) -> (List[str], List[str]):
    clear_all()
    init_space = preprocess_init(init_line_list)
    preprocess_forward(forward_line_list)

    # 具体变异逻辑
    for i in range(len(forward_apis)):
        original_param_kvs.clear()

        # TODO: 简化，init语句暂时不去处理
        # 匹配语句中的self.***，为保证匹配到的是函数，右端匹配到左括号(为止
        # 例如，forward_apis[i]为 x = self.conv1(x), 这里提取 self.conv1
        match = re.search(r'self\.\w+(?=\()', forward_apis[i])

        # 若为注释，或不包含“可变部分”（即self.***）就保持原样
        if forward_apis[i].startswith('#') or forward_apis[i] == '\n' or not match:
            forward_sentence = " " * forward_spaces[i] + forward_apis[i]
            if forward_apis[i] != '\n':
                forward_sentence += "\n"
            forward_output.append(forward_sentence)
            continue

        # 基本信息准备:
        # 1.获取forward中匹配到的self.***部分，例如 self.conv1
        # 2.获取对应的init部分的api（nn.***）, 例如 original_apis["self.conv1"] = "nn.Conv2d(3,2,kernel_size=4,1)"
        # 3.获取api的名字(以nn开头)，例如 nn.Conv2d
        # 4.变为标准api名(以torch开头)，例如 torch.nn.Conv2d
        # 5.参数列表, 例如 [3, 2, kernel_size=4, 1]
        self_name = match.group()
        api_params = original_apis[self_name]
        api_nn = api_params.split('(', 1)[0]
        api_torch = "torch." + api_nn
        api_para_list = api_params.split('(', 1)[1].split(')', -1)[0].split(',')

        # 另立门户，用后缀加以区隔，避免覆盖原有api，例如self.conv1_0
        new_self_name = self_name + "_" + str(init_api_visited[self_name])
        init_api_visited[self_name] += 1

        # 变异后的语句
        init_sentence = " " * init_space + new_self_name + " = "
        forward_sentence = " " * forward_spaces[i] + forward_apis[i].replace(self_name, new_self_name) + "\n"

        # 根据标准api名获取类型，例如 conv
        try:
            category = mapper.get_category(api_torch)
        # 若api不属于任何一类，则保持原状
        except Exception:
            init_sentence += api_params + "\n"
            init_output.append(init_sentence)
            forward_output.append(forward_sentence)
            continue

        # 获取该api的constraint所在yaml文件内容
        content = loader.get_constraint(api_torch)
        param_names = content["inputs"]["required"] + content["inputs"]["optional"]
        for idx, para in enumerate(api_para_list):
            if '=' in para:
                kv_pair = para.split('=')
                original_param_kvs[kv_pair[0].strip()] = kv_pair[1].strip()
            else:
                original_param_kvs[param_names[idx]] = para

        if to_mutate_on_api(category):
            # 两个可能的异常，API Not included 以及 File similarity_avg.yaml needs to be reloaded
            try:
                target_api = simi.get_mutate_api(api_torch)
                content = loader.get_constraint(target_api)
            # 若产生了异常，则保持原样不改变
            except Exception:
                init_sentence += api_params + "\n"
                init_output.append(init_sentence)
                forward_output.append(forward_sentence)
                continue
            api_nn = target_api.split('.', 1)[1]

        init_sentence += api_nn + "("
        mutate_one(content, init_sentence, forward_sentence, category)

    return init_output, forward_output


def mutate_one(content: dict, init_sentence: str, forward_sentence: str, category: str):
    # 提取constraints和inputs信息
    constr = content["constraints"]
    input_list = content["inputs"]

    # 对每个必选参数，若range不是LIMITED则在range范围内变化
    required_list = input_list["required"]
    for idx, item in enumerate(required_list):
        # range为LIMITED, 不变
        if "range" in constr[item].keys() and constr[item]["range"] == "LIMITED":
            val = original_param_kvs[item]
        else:
            val = mutate_logic(constr[item])

        # TODO: linear的特殊处理
        if category == "linear" and item == "in_features":
            spaces = count_space(forward_sentence)
            forward_output.append(" " * spaces + "x = nn.AdaptiveAvgPool1d(" + str(val) + ")(x)\n")

        init_sentence += item + "=" + str(val) + ", "

    optional_list = input_list["optional"]
    for idx, item in enumerate(optional_list):
        # 从前往后变异概率递减
        if random.random() < idx / len(optional_list):
            continue

        # TODO: 这里避免return_indices, groups对于结果的影响
        if item == "return_indices" or item == "groups" or item == "dilation" or item == "padding_idx":
            continue

        val = mutate_logic(constr[item])

        # TODO: 对于dim的处理
        if item == "dim":
            val = -1

        init_sentence += item + "=" + str(val) + ", "

    init_output.append(init_sentence[:-2] + ")\n")
    forward_output.append(forward_sentence)


def mutate_logic(constr: dict):
    # 没有dtype，默认为None
    if "dtype" not in constr.keys():
        return "None"

    default_val = constr["default"] if "default" in constr.keys() else "None"
    ran = constr["range"] if "range" in constr.keys() else "None"
    dtype = constr["dtype"]

    if len(dtype) == 1:
        if dtype[0].startswith("enum"):
            val = '"' + random.choice(ran) + '"'
        else:
            val = random_normal(default_val, dtype[0], ran)
    else:
        type_chosen = random.choice(dtype)
        if type_chosen.startswith("Tuple"):
            inner_type = dtype[1 - dtype.index(type_chosen)]
            match = re.search(r'\((\d+)\)$', type_chosen)
            num = int(match.group(1))

            lst = []
            for _ in range(num):
                item = random_normal(default_val, inner_type, ran)
                lst.append(item)
            val = tuple(lst)
        else:
            val = random_normal(default_val, type_chosen, ran)

    return val


def random_normal(default: str, dtype: str, ran: str):
    val = "None"
    if ran == "None":
        low, high = 0, INF_VAL
    else:
        match1 = re.match(r'\[(.*),\s*(.*)]', str(ran))
        match2 = re.match(r'\[(.*),\s*(.*)\)', str(ran))
        if match1:
            low, high = match1.group(1), match1.group(2)
        elif match2:
            low, high = match2.group(1), match2.group(2)
        else:
            low, high = 1, INF_VAL

        if high == "inf":
            high = INF_VAL
        elif high == "ks/2":
            ks = original_param_kvs["kernel_size"]
            if ',' in ks:
                match = re.match(r'\((\d+), (\d+)\)', ks)
                high = min(int(match.group(1)), int(match.group(2))) // 2
            else:
                high = int(ks) // 2
        # TODO: num_bedding可能存在问题
        elif high == "ne":
            high = 1

    if dtype == "int":
        if default == "None":
            val = random.randint(1, INF_VAL)
        else:
            val = random_normal_int(int(default), low, high)
    elif dtype == "torch.bool":
        val = random.randint(0, 1) == 1
    elif dtype == "torch.float32":
        if default == "None":
            val = random.random() * INF_VAL
        else:
            val = random_normal_float(float(default), low, high)
    return val


def random_normal_int(center, low, high):
    mu = center
    low, high = int(low), int(high)
    sigma = (high - low) / 6
    for i in range(LOOP_TIMES):
        value = int(round(random.gauss(mu, sigma)))
        if low <= value <= high:
            return value
    return center


def random_normal_float(center, low, high):
    mu = center
    low, high = float(low), float(high)
    sigma = (high - low) / 6
    for i in range(LOOP_TIMES):
        value = random.gauss(mu, sigma)
        if low <= value <= high:
            return value
    return center


def clear_all():
    original_apis.clear()
    init_api_visited.clear()
    forward_spaces.clear()
    forward_apis.clear()
    init_output.clear()
    forward_output.clear()


def preprocess_init(init_list: List[str]) -> int:
    for line in init_list:
        if line.strip(' ') == '\n' or line.strip().startswith('#'):
            continue
        lst = line.strip().split("=", 1)

        # 将init部分的每个语句按照“=”拆开，若右侧不为函数API，例如self.a = a（赋值），则保持原状
        if "nn." in lst[1]:
            original_apis[lst[0].strip()] = lst[1].strip()
            init_api_visited[lst[0].strip()] = 0
        else:
            init_output.append(line)

    idx = 0
    while init_list[idx].strip() == '':
        idx += 1
    return count_space(init_list[idx])


def preprocess_forward(forward_list: List[str]) -> None:
    for line in forward_list:
        forward_spaces.append(count_space(line))
        if line.strip(' ') == '\n':
            forward_apis.append("\n")
        else:
            forward_apis.append(line.strip())


# 选择对API进行变异（1）或者对参数进行变异（0）
# TODO: 简化，这里设置linear只能变参数
def to_mutate_on_api(category: str) -> bool:
    return category in ["activation", "rnn"] and random.randint(0, 1) == 1


# 计算空格缩进
def count_space(line: str) -> int:
    space_cnt = 0
    for c in line:
        if c == ' ':
            space_cnt += 1
        elif c == '\t':
            space_cnt += 4
        else:
            break
    return space_cnt


if __name__ == "__main__":
    init_lines = ['\n', '        self.relu = nn.ReLU(inplace=True)\n', '\n',
                  '        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)\n',
                  '        self.avgpool = nn.AdaptiveAvgPool2d(6)\n', '\n',
                  '        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)\n',
                  '        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)\n',
                  '        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)\n',
                  '        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)\n',
                  '        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)\n', '\n',
                  '        self.dropout = nn.Dropout(p=dropout)\n', '\n',
                  '        self.linear1 = nn.Linear(256 * 6 * 6, 4096)\n',
                  '        self.linear2 = nn.Linear(4096, 4096)\n',
                  '        self.linear3 = nn.Linear(4096, num_classes)\n', '\n']

    forward_lines = ['        # 1st block\n', '        x = self.conv1(x)\n', '        x = self.relu(x)\n',
                     '        x = self.pool(x)\n', '\n', '        # 2nd block\n', '        x = self.conv2(x)\n',
                     '        x = self.relu(x)\n', '        x = self.pool(x)\n', '\n', '        # 3rd block\n',
                     '        x = self.conv3(x)\n', '        x = self.relu(x)\n', '\n', '        # 4th block\n',
                     '        x = self.conv4(x)\n', '        x = self.relu(x)\n', '\n', '        # 5th block\n',
                     '        x = self.conv5(x)\n', '        x = self.relu(x)\n', '        x = self.pool(x)\n', '\n',
                     '        # 6th block\n', '        x = self.avgpool(x)\n', '        x = torch.flatten(x, 1)\n',
                     '\n', '        # 7th block\n', '        x = self.dropout(x)\n', '        x = self.linear1(x)\n',
                     '        x = self.relu(x)\n', '\n', '        # 8th block\n', '        x = self.dropout(x)\n',
                     '        x = self.linear2(x)\n', '        x = self.relu(x)\n', '\n', '        # output\n',
                     '        x = self.linear3(x)\n', '\n']

    init_output, forward_output = mutate(init_lines, forward_lines)
    print(init_output)
    print(forward_output)
