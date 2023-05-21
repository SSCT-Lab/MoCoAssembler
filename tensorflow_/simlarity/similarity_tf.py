import json
import operator

import numpy as np
import yaml
import spacy
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from config.paths import tf_func_file, tf_func_def_file, tf_func_param_file, tf_func_sim_file
from utils.Similarity import Similarity


class SimTF(Similarity):

    def __init__(self):
        super().__init__()

    def levenshtein_distance(self, hypothesis: list, reference: list):
        """编辑距离
        计算两个序列的levenshtein distance，可用于计算 WER/CER
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

                    # compare_val = [insertion, deletion, substitution]   # 优先级
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
            # elif ops_matrix[i_idx][j_idx] == 1:   # insert
            elif ops_matrix[i_idx][j_idx] == 2:  # insert
                i -= 1
                nb_map['I'] += 1
            # elif ops_matrix[i_idx][j_idx] == 2:   # delete
            elif ops_matrix[i_idx][j_idx] == 3:  # delete
                j -= 1
                nb_map['D'] += 1
            # elif ops_matrix[i_idx][j_idx] == 3:   # substitute
            elif ops_matrix[i_idx][j_idx] == 1:  # substitute
                i -= 1
                j -= 1
                nb_map['S'] += 1

            # 出边界处理
            if i < 0 and j >= 0:
                nb_map['D'] += 1
            elif j < 0 and i >= 0:
                nb_map['I'] += 1

        match_idx.reverse()
        wrong_cnt = cost_matrix[len_hyp][len_ref]
        nb_map["W"] = wrong_cnt

        return nb_map

    def api_param_sim(self, param_json):
        """
        计算函数参数列表相似度
        :param param_json: 参数列表以字典形式存储在json中
        :return 返回一个字典
        """
        param_sim_dic = {}
        sim = 0.0
        for _ in param_json.items():
            print(_[0], "\033[33mSTART\033[0m")
            for __ in param_json.items():
                if _ == __:
                    sim = 1.0
                else:
                    # 计算参数列表距离
                    nb_map = self.levenshtein_distance(_[1], __[1])
                    if nb_map['N'] != 0:
                        sim = nb_map['C'] / nb_map['N']

                param_sim_dic[__[0]] = sim

            new = Path.open(tf_func_param_file / (_[0] + ".yaml"), "w")
            yaml.dump(param_sim_dic, new)

            print(_[0], "\033[33mDONE\033[0m")

    def text_processing(self, sentence):
        """
        对函数定义的文本进行处理
        :param sentence: 函数定义
        """
        sentence = [token.lemma_.lower()
                    for token in self.nlp(sentence)
                    if not token.is_stop]

        return " ".join(sentence)

    def api_def_sim(self, def_json):
        """
        函数定义相似度计算
        :param def_json: 函数定义以字典形式存储在json中
        """
        def_sim_dic = {}
        if not Path.exists(tf_func_def_file):
            Path.mkdir(tf_func_def_file)

        for _ in def_json.items():
            print(_[0], "\033[31mSTART\033[0m")
            for __ in def_json.items():
                _embedding = self.model.encode(self.text_processing(_[1]), convert_to_tensor=True)
                __embedding = self.model.encode(self.text_processing(__[1]), convert_to_tensor=True)
                cos_score = util.pytorch_cos_sim(_embedding, __embedding)
                sim = cos_score.numpy().tolist()[0]

                def_sim_dic[__[0]] = sim

            new = Path.open(tf_func_def_file / (_[0] + ".yaml"), "w")
            yaml.dump(def_sim_dic, new)
            print(_[0], "\033[31mDONE\033[0m")

    def data_dumps(self):
        # 函数参数列表相似度
        with Path.open(tf_func_file / "param.json", "r") as file:
            data = json.load(file)
            self.api_param_sim(data)

        # 函数定义相似度
        with Path.open(tf_func_file / "def.json", "r") as file:
            data = json.load(file)
            self.api_def_sim(data)

    def sim_calcu(self, w_def: float, w_param: float):

        if not Path.exists(tf_func_sim_file):
            Path.mkdir(tf_func_sim_file)

        with Path.open(tf_func_file / "def.json", "r") as file:
            data = json.load(file)
        api_list = list(data.keys())

        for api in api_list:
            sim = {}
            print(api, "\033[34mSTART\033[0m")
            def_file = Path.open(tf_func_def_file / (api + ".yaml"), "r")
            param_file = Path.open(tf_func_param_file / (api + ".yaml"), "r")

            def_data = yaml.load(def_file, yaml.Loader)
            param_data = yaml.load(param_file, yaml.Loader)

            for _ in api_list:
                def_sim = def_data[_][0]
                param_sim = param_data[_]
                sim[_] = def_sim * w_def + param_sim * w_param

            new = Path.open(tf_func_sim_file / (api + ".yaml"), "w")
            res = dict(sorted(sim.items(), key=operator.itemgetter(1), reverse=True))
            yaml.dump(res, new, sort_keys=False)
            print(api, "\033[34mDONE\033[0m")


if __name__ == "__main__":
    # 写数据
    # data_dumps()

    simTF = SimTF()
    # 根据权重中计算相似度
    w_def = 0.5
    w_param = 0.5
    simTF.sim_calcu(w_def, w_param)
