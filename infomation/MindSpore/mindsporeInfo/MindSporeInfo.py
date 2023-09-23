import os
import re
import yaml
from utils import Info, Param

mindspore_nn_apis = 'https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.nn.html'
mindspore_nn_api = 'https://www.mindspore.cn/docs/en/r2.1/api_python/nn/'
headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'}

import requests
from bs4 import BeautifulSoup


def get_apis() -> list:
    response = requests.get(mindspore_nn_apis, headers=headers)
    content = response.text
    soup = BeautifulSoup(content, 'html.parser')

    apis = soup.select('tr > td > p > a > code > span')

    response.close()

    api_list = [api.text for api in apis if api.text.startswith('mindspore.nn')]
    return api_list


def get_infos(api):
    optional = []
    required = []
    param_list = []

    api_html = f'{mindspore_nn_api}{api}.html'
    response = requests.get(api_html, headers=headers)
    content = response.text
    soup = BeautifulSoup(content, 'html.parser')

    dt = soup.select_one('section > dl > dt')
    api = dt.text.strip()[:-1].replace('class ', '').replace('[source]', '')
    api_descp = re.findall(r'(.*?\.).*?', soup.select_one('dd > p').text, re.S)[0].replace('\n', ' ')

    try:
        dd = soup.find_all('dd', class_='field-odd')[0]
    except IndexError as e:
        print(f'{api}中找不到dd标签，可能是无参数')
        return None

    ps = dd.find_all('p')
    for p in ps:
        strong = p.find('strong')
        if strong:
            name = strong.text
            param_descp = p.text.replace('\n', ' ')
            default = re.findall(r'.*?Default:(.*?)\..*?', param_descp, re.S)
            default = default[0].strip() if default else None
            try:
                dtype = re.findall(r'.*? \((.*?)\) .*?', param_descp, re.S)[0]
            except IndexError as e:
                print(f'{api}中的{name}参数没有dtype')
                return None

            param = Param(name, param_descp, default, dtype)
            param_list.append(param)

            if 'optional' in dtype:
                optional.append(name)
            elif 'required' in dtype:
                required.append(name)

    info = Info(api, api_descp, param_list, optional, required)
    response.close()
    return info.dict_info()


def yaml_dumps(infos, api):
    file_path = 'infos'
    try:
        os.makedirs(file_path)
    except:
        pass

    file_name = f'{file_path}/{api}.yaml'

    with open(file_name, 'w', encoding='utf-8') as yaml_file:
        yaml.safe_dump(infos, yaml_file, allow_unicode=True, sort_keys=False)


if __name__ == '__main__':
    apis = get_apis()
    for api in apis:
        infos = get_infos(api)
        if infos:
            yaml_dumps(infos, api)

    # infos = get_infos('mindspore.nn.Conv2d')
    # yaml_dumps(infos, 'mindspore.nn.Conv2d')

    # with open('/Users/wuduo/Documents/BioWork/DifferentialTestingofDLFrameworks/mindsporeInfo/infos/mindspore.nn.Conv2d.yaml', 'r', encoding='utf-8') as file:
    #     data = yaml.load(file, yaml.Loader)
    #     constraints = data['constraints']
    #     for param in constraints.items():
    #         param_ = param[1]
    #         print(f'{param[0]}\ntype of descp: {type(param_["descp"])}\n descp: {param_["descp"]}')
