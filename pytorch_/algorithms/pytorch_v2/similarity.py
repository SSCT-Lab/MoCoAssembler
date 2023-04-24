import numpy as np
import os
import yaml

import marker
import category_mapper as mapper
import utils

import sentence_transformers as st
import sentence_transformers.util

model = st.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

constraints_data = {}


# 计算并保存相似度的整体流程
def calculate_similarity() -> None:
    category_list = os.listdir(marker.CONSTRAINTS_PATH)
    for category in category_list:
        cur_path = os.path.join(marker.CONSTRAINTS_PATH, category)
        file_list = os.listdir(cur_path)

        for file in file_list:
            if "torch.nn." not in file:
                continue
            with open(os.path.join(cur_path, file), "r", encoding="utf-8") as f:
                content = yaml.full_load(f.read())
                f.close()
            constraints_data[file[:-5]] = content

        print("Calculate similarity on", category, "starts...")
        calculate_def_similarity(path=cur_path)
        calculate_para_similarity(path=cur_path)
        calculate_average_similarity(path=cur_path)
        constraints_data.clear()
        print("Calculate similarity on", category, "ends...")


# 计算参数列表相似度，调用levenshtein_distance
def calculate_para_similarity(path: str) -> None:
    api_list = list(constraints_data.keys())
    para_list = []
    for api in api_list:
        cur_api = constraints_data[api]['api']
        para_temp_list = cur_api.split("(")[1].split(")")[0].split(",")
        para_list.append(para_temp_list)

    simi_dict = {}
    for i in range(len(api_list)):
        simi_temp_dict = {}
        for j in range(len(api_list)):
            if i != j:
                simi_temp_dict[api_list[j]] = round(utils.levenshtein_distance(para_list[i], para_list[j]), 4)
        simi_dict[api_list[i]] = simi_temp_dict

    with open(os.path.join(path, "similarity_para.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(simi_dict, f, allow_unicode=True)


# 计算函数定义相似度，使用文本的余弦相似度
def calculate_def_similarity(path: str) -> None:
    api_list = list(constraints_data.keys())
    descp_list = []
    for i in range(len(api_list)):
        descp = constraints_data[api_list[i]]['descp']
        descp_list.append(utils.text_processing(descp))

    api_num = len(api_list)
    simi_dict = {}

    corpus_embeddings = model.encode(descp_list, convert_to_tensor=True)

    for i in range(api_num):
        sentence_embedding = model.encode(descp_list[i], convert_to_tensor=True)
        scores = sentence_transformers.util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
        top_results = np.argpartition(-scores, range(api_num)).tolist()

        simi_temp_dict = {}
        for idx in top_results[0:api_num]:
            if idx != i:
                simi_temp_dict[api_list[idx]] = round(scores[idx].item(), 4)
        simi_dict[api_list[i]] = simi_temp_dict

    with open(os.path.join(path, "similarity_descp.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(simi_dict, f, allow_unicode=True)
        f.close()


# 总体相似度，目前直接将上面两者平均
def calculate_average_similarity(path: str) -> None:
    with open(os.path.join(path, "similarity_descp.yaml"), "r", encoding="utf-8") as f:
        descp_simi = yaml.full_load(f.read())
        f.close()
    with open(os.path.join(path, "similarity_para.yaml"), "r", encoding="utf-8") as f:
        para_simi = yaml.full_load(f.read())
        f.close()

    for api1 in descp_simi.keys():
        for api2 in descp_simi[api1].keys():
            descp_simi[api1][api2] = round((descp_simi[api1][api2] + para_simi[api1][api2]) / 2, 4)

    with open(os.path.join(path, "similarity_avg.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(descp_simi, f, allow_unicode=True)
        f.close()


# 根据相似度随机选取同类API
def get_mutate_api(api: str) -> str:
    target = ""

    try:
        category = mapper.get_category(api)
        simi_path = os.path.join(marker.CONSTRAINTS_PATH, category, "similarity_avg.yaml")
        with open(simi_path, "r", encoding="utf-8") as f:
            content = yaml.full_load(f.read())
            f.close()

        if api not in content.keys():
            raise Exception(category + "/similarity_avg.yaml doesn't contain" + api + ", which needs to be reloaded")

        if category == "conv" or category == "pooling":
            target = utils.select_api_conv_pool(content[api], api)
        else:
            target = utils.select_api_by_probability(content[api])

    except Exception as err:
        print(err)

    return target


if __name__ == '__main__':
    calculate_similarity()
    # print(get_mutate_api("torch.nn.ReLU"))
