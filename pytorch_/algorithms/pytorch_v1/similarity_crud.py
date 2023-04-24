import os
import yaml
from typing import Dict

import marker

'''
读取similarity
read_similarity: 读取similarity，接口，有两种实现
    read_similarity_yaml: 读取所在的yaml文件信息
    read_similarity_mysql: 读取所在的mysql表信息
'''


def read_similarity(api: str) -> Dict[str, int]:
    similarity_dict = read_similarity_yaml(api)
    # similarity_dict = read_similarity_mysql(api)
    return similarity_dict


def read_similarity_yaml(api: str) -> Dict[str, int]:
    file = get_similarity_yaml(api)
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        content = yaml.full_load(content)
        f.close()

    similarity_dict = {}
    if content is not None and api in content.keys():
        similarity_dict = content[api]

    return similarity_dict


# TODO: MySQL实现
def read_similarity_mysql(api: str) -> Dict[str, int]:
    return {}


'''
更新similarity
update_similarity: 更新similarity，接口，有两种实现
    update_similarity_yaml: 更新所在的yaml文件信息
    update_similarity_mysql: 更新所在的mysql表信息
'''


def update_similarity(api_0: str, api_1: str, similarity: float) -> None:
    update_similarity_yaml(api_0, api_1, similarity)
    # update_similarity_mysql(api_0, api_1, similarity)


# 使用yaml文件保存similarity的信息
def update_similarity_yaml(api_0: str, api_1: str, similarity: float) -> None:
    file = get_similarity_yaml(api_0)

    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        content = yaml.full_load(content)
        f.close()

    content[api_0][api_1] = similarity
    content[api_1][api_0] = similarity

    with open(file, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True)

    print("update success")


# TODO: MySQL实现
def update_similarity_mysql(api_0: str, api_1: str, similarity: float) -> None:
    return


# 查询api所在similarity的文件路径
def get_similarity_yaml(api: str) -> str:
    file = os.path.join(marker.ROOT_PATH, "constraints", "category_mapping.yaml")
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        content = yaml.full_load(content)
        f.close()

    if api in content.keys():
        category = content[api]
    else:
        category = "zoo"  # 称为zoo目的是放在pytorch_modified的最后一个

    file = os.path.join(marker.ROOT_PATH, "constraints", "pytorch_modified", category, "similarity.yaml")
    return file


if __name__ == '__main__':
    # update_similarity("torch.nn.AdaptiveMaxPool1d", "torch.nn.AdaptiveMaxPool2d", 0.95)
    print(read_similarity("torch.nn.AdaptiveMaxPool1d"))
