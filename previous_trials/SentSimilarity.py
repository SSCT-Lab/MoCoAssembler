import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
requests.adapters.DEFAULT_RETRIES = 5
nlp = spacy.load("en_core_web_sm")
headers = {'Connection': 'close',}
s = requests.session()
s.keep_alive = False

def levenshtein_distance(hypothesis: list, reference: list):
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
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1

                # compare_val = [insertion, deletion, substitution]   # 优先级
                compare_val = [substitution, insertion, deletion]   # 优先级

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

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1

            # 出边界后，这里仍然使用，应为第一行与第一列必然是全零的
            i -= 1
            j -= 1
        # elif ops_matrix[i_idx][j_idx] == 1:   # insert
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        # elif ops_matrix[i_idx][j_idx] == 2:   # delete
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        # elif ops_matrix[i_idx][j_idx] == 3:   # substitute
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
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

    # print("ref: %s" % " ".join(reference))
    # print("hyp: %s" % " ".join(hypothesis))
    # print(nb_map)
    # print("match_idx: %s" % str(match_idx))
    # return wrong_cnt, match_idx, nb_map
    return nb_map



def text_processing(sentence):
    """
    Lemmatize, lowercase, remove numbers and stop words

    Args:
      sentence: The sentence we want to process.

    Returns:
      A list of processed words
    """
    sentence = [token.lemma_.lower()
                for token in nlp(sentence)
                if token.is_alpha and not token.is_stop]

    return " ".join(sentence)


def get_api_parameters(url):
    resp = s.get(url)
    resp.encoding = "UTF-8"  # 处理乱码
    main_page = BeautifulSoup(resp.text, "html.parser")
    dd_elems = main_page.find_all(name='dd', attrs={'class': 'field-odd'})
    key_elems = main_page.find_all(name='dd', attrs={'class': 'field-even'})
    para_list = []
    if len(dd_elems) > 0:
        if dd_elems[0].next.name == "p":
            para_list.append(dd_elems[0].text.replace("\n", " ").strip())
        elif dd_elems[0].next.name == "ul":
            dd_elems = dd_elems[0].contents[0].contents
            for dd in dd_elems:
                if dd != "\n":
                    para_list.append(dd.text.replace("\n", " ").strip())

    if len(key_elems) > 0:
        if key_elems[0].next.name == "p":
            para_list.append(key_elems[0].text.replace("\n", " ").strip())
        elif key_elems[0].next.name == "ul":
            key_elems = key_elems[0].contents[0].contents
            for dd in key_elems:
                if dd != "\n":
                    para_list.append(dd.text.replace("\n", " ").strip())

    return para_list


def get_api_definition(main_page, pkg, w):
    table_list = main_page.find_all(name='table', attrs={'class': 'autosummary longtable docutils align-default'})
    api_json = {}
    for table in table_list:
        row_data = []
        parent_node = table.parent
        api_class = parent_node.attrs['id']
        rows = table.find_all('tr')
        for row in rows:
            td_cols = row.find_all(name='td')
            if len(td_cols) > 0:
                api_name = td_cols[0].text.strip()
                print(api_name)
                definition = td_cols[1].text.strip()
                print(definition)
                if len(row.find_all(name='a')) > 0:
                    para_html = "https://pytorch.org/docs/stable/" + row.find_all(name='a')[0].attrs['href']
                    para_list = get_api_parameters(para_html)
                    print(para_list)
                else:
                    para_list = []
                row_data.append({"api_name": api_name, "definition": definition, "parameters": para_list})
        api_json[api_class] = row_data

    return api_json


def calculate_sent_simi(sent1, sent2):
    # Compute embedding for both lists
    embedding_1 = model.encode(sent1, convert_to_tensor=True)
    embedding_2 = model.encode(sent2, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    print(sim)
    return sim


def calculate_api_sim(api_json, pkg):
    w = open("./api_similarity/" + pkg + ".txt", "w")
    for key in api_json.keys():
        api_list = api_json[key]
        w.write("api-function : " + key + "\n")
        for i in range(len(api_list) - 1):
            def1 = text_processing(api_list[i]['definition'])
            other_apis = api_list[i + 1:]
            corpus = [text_processing(item['definition']) for item in other_apis]
            if len(corpus) > 0:
                corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
                sentence_embedding = model.encode(def1, convert_to_tensor=True)
                # compute similarity scores of the sentence with the corpus
                top_k = len(corpus)
                cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
                top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
                print("Sentence:", def1, "\n")
                print("Top", top_k, "most similar sentences in corpus:")
                # w.write("API : " + api_list[i]['api_name'] + " | Definition : " + api_list[i]['definition'] + "\n")
                for idx in top_results[0:top_k]:
                    if cos_scores[idx].item() > 0:
                        print(corpus[idx], "(Score: %.4f)" % (cos_scores[idx].item()))
                        w.write(api_list[i]['api_name'] + " | " + api_list[i]['definition'] + " | " + other_apis[idx]['api_name'] + " | " + other_apis[idx]['definition'] + " | " + str(cos_scores[idx].item()) + "\n")

        w.write("\n")
    w.close()


def load_api_parameters():
    f = open("./api_parameters/torch_api.txt", "r")
    api_para = {}
    line = f.readline()
    while line:
        if "API-function:" in line:
            api_type = line[:-1].replace("API-function: ", "")
            api_para[api_type] = {}
        if "API:" in line:
            api = line[:-1].replace("API:", "")
            api_para[api_type][api] = []
            line = f.readline()
            while line != "\n":
                api_para[api_type][api].append(line[:-1].split(" – ")[0])
                line = f.readline()
        line = f.readline()

    return api_para


def obtain_para_sim(api_para):
    file = open("./para_sim.txt", "w")
    for f in api_para.keys():
        file.write("API-function: " + f + "\n")
        api_dict = api_para[f]
        api_name = list(api_dict.keys())
        for i in range(len(api_name) - 1):
            #sim_list = []
            for j in range(i+1, len(api_name)):
                nb_map = levenshtein_distance(api_dict[api_name[i]], api_dict[api_name[j]])
                if nb_map['N'] != 0:
                    wcr = nb_map['C']/nb_map['N']
                    file.write(api_name[i] + " | " + api_name[j] + " | " + str(wcr) + "\n")

        file.write("\n")

    file.close()


def load_sim_file(path):
    f = open(path, "r")
    line = f.readline()
    api_sim = {}
    while line:
        if "API-function: " in line:
            api_type = line[:-1].replace("API-function: ", "")
            api_sim[api_type] = []
            line = f.readline()
            while line != "\n":
                line = line[:-1].split(" | ")
                if len(line) > 3:
                    line.remove(line[1])
                    line.remove(line[2])
                    api_sim[api_type].append(line)
                else:
                    api_sim[api_type].append(line)
                line = f.readline()
        line = f.readline()

    return api_sim


if __name__ == '__main__':
    # url = "https://pytorch.org/docs/stable/torch.html"
    # resp = s.get(url)
    # resp.encoding = "UTF-8"  # 处理乱码
    # main_page = BeautifulSoup(resp.text, "html.parser")
    # # pkg_list = main_page.find_all(name='ul', attrs={'class': 'current'})[0].text.strip().split("\n")
    # # ahref_list =
    # pkg_list = main_page.find_all(name='ul', attrs={'class': 'current'})[0]
    # pkg_name = []
    # ahref_list = []
    # for pkg in pkg_list:
    #     if pkg != "\n":
    #         a = pkg.next.attrs['href']
    #         if a == "#":
    #             a = "torch.html"
    #         pkg_name.append(pkg.text.strip())
    #         ahref_list.append(a)
    #
    # w = open("./api_parameters/torch_api.txt", "w")
    # for i in range(len(pkg_name)):
    #     url = "https://pytorch.org/docs/stable/" + ahref_list[i]
    #     resp = s.get(url)
    #     resp.encoding = "UTF-8"  # 处理乱码
    #     main_page = BeautifulSoup(resp.text, "html.parser")
    #     api_json = get_api_definition(main_page, pkg_name[i], w)
    #     #calculate_api_sim(api_json, pkg_name[i])
    #     for key in api_json.keys():
    #         w.write("API-function: " + pkg_name[i] + "-" + key + "\n")
    #         api_list = api_json[key]
    #         for item in api_list:
    #             w.write('API:' + item['api_name']+ "\n")
    #             para_list = item['parameters']
    #             for para in para_list:
    #                 w.write(para + "\n")
    #             w.write("\n")
    # w.close()
    def_sim = load_sim_file("../api_similarity/torch_definition_sim.txt")
    para_sim = load_sim_file("../api_similarity/torch_para_sim.txt")

