import os
import random
import re
import yaml

from config import marker

INF_VAL, LOOP_TIMES = 4, 10


def get_mutate_api(api: str) -> str:
    target = ""

    try:
        category = get_category(api)
        simi_path = os.path.join(marker.CONSTRAINTS_PATH, category, "similarity_avg.yaml")
        with open(simi_path, "r", encoding="utf-8") as f:
            content = yaml.full_load(f.read())
            f.close()

        if api not in content.keys():
            raise Exception(category + "/similarity_avg.yaml doesn't contain" + api + ", which needs to be reloaded")

        if category == "conv" or category == "pooling":
            target = select_api_conv_pool(content[api], api)
        else:
            target = select_api_by_probability(content[api])

    except Exception as err:
        print(err)

    return target


def get_category(api: str) -> str:
    with open(marker.MAPPING_FILE, "r", encoding="utf-8") as f:
        content = yaml.full_load(f.read())
        f.close()

    if api not in content.keys():
        raise Exception("API " + api + " not included")

    return content[api]


def mutate_logic(constr: dict):
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


def to_mutate_on_api(category: str) -> bool:
    return category in ["activation", "rnn"] and random.randint(0, 1) == 1


def select_api_by_probability(prob_dict) -> str:
    rand_num = random.uniform(0, sum(prob_dict.values()))
    prob_sum = 0

    for k, v in prob_dict.items():
        prob_sum += v
        if rand_num < prob_sum:
            return k


def select_api_conv_pool(prob_dict, api) -> str:
    degree = int(re.search(r'\d+', api).group())
    new_dict = {}
    for k, v in prob_dict.items():
        degree_temp = int(re.search(r'\d+', k).group())
        if degree_temp == degree:
            new_dict[k] = v
    return select_api_by_probability(new_dict)


def random_normal(default: str, dtype: str, ran: str):
    val = "None"
    if ran == "None":
        low, high = 1, INF_VAL
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
            high = INF_VAL
            # TODO: kernel_size
            # ks = original_param_kvs["kernel_size"]
            # if ',' in ks:
            #     match = re.match(r'\((\d+), (\d+)\)', ks)
            #     high = min(int(match.group(1)), int(match.group(2))) // 2
            # else:
            #     high = int(ks) // 2
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
