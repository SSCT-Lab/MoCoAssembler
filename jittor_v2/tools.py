import random


# 轮盘赌算法，按照相对概率均匀选择相似API
def select_api_by_probability(prob_dict) -> str:
    rand_num = random.uniform(0, sum(prob_dict.values()))
    prob_sum = 0

    for k, v in prob_dict.items():
        prob_sum += v
        if rand_num < prob_sum:
            return k


def random_select_from_list(lst):
    if not lst:
        return None
    return random.choice(lst)


def get_string_type(s):
    if s == 'None':
        return 'int'
    if s.lower() == 'true' or s.lower() == 'false':
        return 'bool'
    try:
        num = int(s)
        return 'int'
    except ValueError:
        try:
            num = float(s)
            return 'float'
        except ValueError:
            if s.startswith('[') and s.endswith(']'):
                return 'list'
            elif s.startswith('(') and s.endswith(')'):
                return 'tuple'
            elif s.startswith('{') and s.endswith('}'):
                return 'dict'
            else:
                return 'str'


def roulette_wheel_selection(prob_dict):
    """
    轮盘赌算法选择一个元素
    :param prob_dict: dict，元素及其对应的概率
    :return: 选择的元素
    """
    sum_prob = sum(prob_dict.values())  # 计算概率总和
    rand = random.uniform(0, 1)  # 生成一个0到1之间的随机数
    proportion_list = [prob / sum_prob for prob in prob_dict.values()]  # 计算每个元素的比例
    dist_list = [abs(proportion_list[i] - rand) for i in range(len(proportion_list))]  # 计算每个比例与随机数之间的距离
    min_dist_index = dist_list.index(min(dist_list))  # 找到距离最小的比例所对应的元素
    return list(prob_dict.keys())[min_dist_index]
