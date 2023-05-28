import random
import re
import yaml
from config import marker

LOOP_TIMES = 10


def get_category(api: str) -> str:
    with open(marker.MAPPING_FILE, "r", encoding="utf-8") as f:
        content = yaml.full_load(f.read())
        f.close()

    if api not in content.keys():
        raise Exception("API " + api + " not included")

    return content[api]


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
