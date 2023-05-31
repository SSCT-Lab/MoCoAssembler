import os
import re

import numpy as np
import spacy
import yaml
from sentence_transformers import SentenceTransformer

import en_core_web_sm
import sentence_transformers.util

from utils.Similarity import Similarity
from config import marker


class SimilarityPT(Similarity):

    def __init__(self):
        super().__init__()
        # 这里我的环境中只能使用import的方式导入模型，若无法运行则替换为原句 nlp = spacy.load("en_core_web_sm")
        self.nlp = en_core_web_sm.load()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.file_lst = []
        self.def_lst = []
        self.param_lst = []

    def data_dumps(self):
        file_lst = os.listdir(marker.SIMILARITY_PATH)
        self.file_lst = file_lst

        for file in file_lst:
            file_fullname = os.path.join(marker.SIMILARITY_PATH, file)
            with open(file_fullname, "r", encoding="utf-8") as f:
                content = yaml.full_load(f.read())
                f.close()
            self.def_lst.append(self.text_processing(content["descp"]))
            self.param_lst.append(re.findall(r'\((.*?)\)', content["api"]))

        print("Calculation on definition similarity begins...")
        self.api_def_sim("")
        print("Calculation on definition similarity ends...")

        print("Calculation on parameter-list similarity begins...")
        self.api_param_sim("")
        print("Calculation on parameter-list similarity ends...")

        print("Calculation on average similarity begins...")
        self.sim_calcu(0.5, 0.5)
        print("Calculation on average similarity ends...")

    def api_def_sim(self, def_json):
        corpus_embeddings = self.model.encode(self.def_lst, convert_to_tensor=True)

        api_num = len(self.file_lst)
        for i in range(api_num):
            sentence_embedding = self.model.encode(self.def_lst[i], convert_to_tensor=True)
            scores = sentence_transformers.util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
            top_results = np.argpartition(-scores, range(api_num)).tolist()

            simi_dict = {}
            for idx in top_results[0:api_num]:
                simi_dict[self.file_lst[idx][:-5]] = round(scores[idx].item(), 4)

            def_simi_target = os.path.join(marker.DEF_SIMI_TARGET_PATH, self.file_lst[i])
            with open(def_simi_target, "w", encoding="utf-8") as f:
                yaml.dump(simi_dict, f, allow_unicode=True)
                f.close()

            print(self.file_lst[i], i, "finished")

    def api_param_sim(self, param_json):
        api_num = len(self.file_lst)
        for i in range(api_num):
            simi_dict = {}
            for j in range(api_num):
                dist = self.levenshtein_distance(self.param_lst[i], self.param_lst[j])
                simi_dict[self.file_lst[j][:-5]] = round(dist, 4)

            param_simi_target = os.path.join(marker.PARAM_SIMI_TARGET_PATH, self.file_lst[i])
            with open(param_simi_target, "w", encoding="utf-8") as f:
                yaml.dump(simi_dict, f, allow_unicode=True)
                f.close()

            print(self.file_lst[i], i, "finished")

    def sim_calcu(self, w_def: float, w_param: float):
        api_num = len(self.file_lst)
        for i in range(api_num):
            def_simi_target = os.path.join(marker.DEF_SIMI_TARGET_PATH, self.file_lst[i])
            with open(def_simi_target, "r", encoding="utf-8") as f:
                descp_simi = yaml.full_load(f.read())
                f.close()

            param_simi_target = os.path.join(marker.PARAM_SIMI_TARGET_PATH, self.file_lst[i])
            with open(param_simi_target, "r", encoding="utf-8") as f:
                para_simi = yaml.full_load(f.read())
                f.close()

            for api in descp_simi.keys():
                descp_simi[api] = round(descp_simi[api] * w_def + para_simi[api] * w_param, 4)

            avg_simi_target = os.path.join(marker.SIMI_TARGET_PATH, self.file_lst[i])
            with open(avg_simi_target, "w", encoding="utf-8") as f:
                yaml.dump(descp_simi, f, allow_unicode=True)
                f.close()

            print(self.file_lst[i], i, "finished")

    def text_processing(self, sentence):
        sentence = [token.lemma_.lower()
                    for token in self.nlp(sentence)
                    if token.is_alpha and not token.is_stop]
        return " ".join(sentence)

    def levenshtein_distance(self, hypothesis: list, reference: list):
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


if __name__ == "__main__":
    simi = SimilarityPT()
    simi.data_dumps()
