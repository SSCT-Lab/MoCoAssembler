import numpy as np
import os
import yaml

import file_paths
import tools

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers


class Similarity:
    def __init__(self):
        self.api_path = file_paths.API_INFO_PATH
        self.all_api_info = {}
        self.get_all_api_info()
        self.model_to_use = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def set_path(self, path):
        self.api_path = path
        self.all_api_info.clear()
        self.get_all_api_info()

    def get_all_api_info(self) -> None:
        path = self.api_path
        file_list = os.listdir(path)

        for file in file_list:
            if "jittor.nn." not in file:
                continue
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                content = yaml.full_load(f.read())
                f.close()
            self.all_api_info[file[:-5]] = content

    def get_api_list(self) -> list:
        return list(self.all_api_info.keys())

    def calculate_description_similarity(self, api_name_1: str, api_name_2: str, model):
        description_1, description_2 = self.all_api_info[api_name_1]['descp'], self.all_api_info[api_name_2]['descp']
        # vectorizer = TfidfVectorizer()
        # vectorizer2 = CountVectorizer()
        if description_1.startswith('no') or description_2.startswith('no'):
            return tools.calculate_cosine_similarity(api_name_1.split('.')[2], api_name_2.split('.')[2], model)
        description_1, description_2 = tools.text_processing(description_1), tools.text_processing(description_2)
        return tools.calculate_cosine_similarity(description_1, description_2, model)

        # description_1, description_2 = description_1.split(), description_2.split()
        # vectorizer_text = vectorizer.fit_transform([description_1, description_2])
        # similarity = cosine_similarity(vectorizer_text)
        # result = round(
        #     0.15 * similarity[0][0] + 0.4 * similarity[0][1] + 0.35 * similarity[1][0] + 0.1 * similarity[1][1], 4)
        # if result >= 0.9998:
        #     return 1
        # else:
        #     return result

    def calculate_parameter_similarity(self, api_name_1: str, api_name_2: str):
        api_1, api_2 = self.all_api_info[api_name_1]['api'], self.all_api_info[api_name_2]['api']
        para_list_1, para_list_2 = api_1.split('(')[1].split(')')[0].split(","), api_2.split('(')[1].split(')')[
            0].split(",")
        return round(tools.levenshtein_distance(para_list_1, para_list_2), 4)

    def calculate_similarity_list(self) -> dict:
        api_list = self.get_api_list()
        api_num = len(api_list)
        similarity_dictionary = {}
        model = self.model_to_use
        for i in range(api_num):
            temp_dictionary = {}
            for j in range(api_num):
                if i != j:
                    descp_similarity = self.calculate_description_similarity(api_list[i], api_list[j], model)
                    param_similarity = self.calculate_parameter_similarity(api_list[i], api_list[j])
                    temp_dictionary[api_list[j]] = float(round(descp_similarity * 0.5 + param_similarity * 0.5, 4))
            similarity_dictionary[api_list[i]] = temp_dictionary
        return similarity_dictionary


if __name__ == '__main__':
    s_api, s_layer = Similarity(), Similarity()
    s_layer.set_path(file_paths.LAYER_INFO_PATH)
    f1 = open(os.path.join(file_paths.API_SIMILARITY_PATH, 'api_similarity.yaml'), 'w')
    f2 = open(os.path.join(file_paths.LAYER_SIMILARITY_PATH, 'layer_similarity.yaml'), 'w')
    dic_api = s_api.calculate_similarity_list()
    dic_layer = s_layer.calculate_similarity_list()
    yaml.dump(dic_api, f1)
    yaml.dump(dic_layer, f2)
    f1.close()
    f2.close()
