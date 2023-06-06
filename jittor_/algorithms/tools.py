import random
import re
import numpy as np
# import torch
import spacy
import sentence_transformers
import sentence_transformers.util


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


# TODO: 简化，这里conv和pooling遇到维度问题只能同维度变化
def select_api_conv_pool(prob_dict, api) -> str:
    degree = int(re.search(r'\d+', api).group())
    new_dict = {}
    for k, v in prob_dict.items():
        degree_temp = int(re.search(r'\d+', k).group())
        if degree_temp == degree:
            new_dict[k] = v
    return select_api_by_probability(new_dict)


# 语句处理: 只保留char组成的语词，并去除停词组成一个句子
def text_processing(sentence: str) -> str:
    nlp = spacy.load('en_core_web_sm')
    sentence = [token.lemma_.lower()
                for token in nlp(sentence)
                if token.is_alpha and not token.is_stop]
    return " ".join(sentence)


# 编辑距离: 计算两个序列的levenshtein distance，可用于计算 WER/CER
def levenshtein_distance(hypothesis: list, reference: list):
    """
    参考资料：
        https://www.cuelogic.com/blog/the-levenshtein-algorithm
        https://martin-thoma.com/word-error-rate-calculation/

    C: correct
    W: wrong
    I: insert
    D: delete
    S: substitution

    :param hypothesis: 预测序列
    :param reference: 真实序列
    :return: 1: 错误操作，所需要的 S，D，I 操作的次数;
             2: ref 与 hyp 的所有对齐下标
             3: 返回 C、W、S、D、I 各自的数量
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    # 记录所有的操作，0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    # 生成 cost 矩阵和 operation矩阵，i:外层hyp，j:内层ref
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i - 1] == reference[j - 1]:
                cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
            else:
                substitution = cost_matrix[i - 1][j - 1] + 1
                insertion = cost_matrix[i - 1][j] + 1
                deletion = cost_matrix[i][j - 1] + 1

                compare_val = [substitution, insertion, deletion]  # 优先级

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []  # 保存 hyp与ref 中所有对齐的元素下标
    i = len_hyp
    j = len_ref

    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}

    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:  # correct
            if i - 1 >= 0 and j - 1 >= 0:
                match_idx.append((j - 1, i - 1))
                nb_map['C'] += 1
            # 出边界后，这里仍然使用，应为第一行与第一列必然是全零的
            i -= 1
            j -= 1

        elif ops_matrix[i_idx][j_idx] == 1:  # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        elif ops_matrix[i_idx][j_idx] == 2:  # insert
            i -= 1
            nb_map['I'] += 1

        elif ops_matrix[i_idx][j_idx] == 3:  # delete
            j -= 1
            nb_map['D'] += 1

        # 出边界处理
        if i < 0 <= j:
            nb_map['D'] += 1
        elif j < 0 <= i:
            nb_map['I'] += 1

    wcr = nb_map['C'] / nb_map['N']
    return wcr


def cosine_similarity(str1, str2):
    # 构建字符集合
    chars = set(str1).union(set(str2))
    # 构建字符出现次数向量
    vec1 = [str1.count(char) for char in chars]
    vec2 = [str2.count(char) for char in chars]
    # 计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def calculate_cosine_similarity(description_1: str, description_2: str, model):
    # model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    d_1 = model.encode(description_1, convert_to_tensor=True)
    d_2 = model.encode(description_2, convert_to_tensor=True)
    score = sentence_transformers.util.pytorch_cos_sim(d_1, d_2)
    return round(score.item(), 4)
