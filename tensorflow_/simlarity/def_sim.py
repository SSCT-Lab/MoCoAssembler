# -*- coding: utf-8 -*-

"""
@Title   : 函数定义相似度计算
@Time    : 2023/4/14 13:36
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""

import json
import yaml
import spacy
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from config.paths import tf_func_file, tf_func_def_file

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def text_processing(sentence):
    """
    Lemmatize, lowercase, remove numbers and stop words
    :param sentence: The sentence we want to process.
    :return: A list of processed words
    """
    sentence = [token.lemma_.lower()
                for token in nlp(sentence)
                if not token.is_stop]

    return " ".join(sentence)

def api_def_sim(def_json):
    """

    :param api_json: 输入json格式的数据
    :return: 写入yaml文件
    """
    def_sim_dic = {}

    if not Path.exists(tf_func_def_file):
        Path.mkdir(tf_func_def_file)

    for _ in def_json.items():
        print(_[0], "\033[33mSTART\033[0m")
        for __ in def_json.items():
            _embedding = model.encode(text_processing(_[1]), convert_to_tensor=True)
            __embedding = model.encode(text_processing(__[1]), convert_to_tensor=True)
            cos_score = util.pytorch_cos_sim(_embedding, __embedding)
            sim = cos_score.numpy().tolist()[0]

            def_sim_dic[__[0]] = sim

        new = Path.open(tf_func_def_file / (_[0]+ ".yaml"), "w")
        yaml.dump(def_sim_dic, new)
        print(_[0], "\033[31mDONE\033[0m")


if __name__ == "__main__":
    with open(tf_func_file / "def.json", "r") as file:
        data = json.load(file)
        api_def_sim(data)
