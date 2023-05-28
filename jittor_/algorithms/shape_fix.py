# 5.28: shape fix
import os.path
import random

import yaml
import file_paths


class ShapeFixer:
    def __init__(self):
        f = open(os.path.join(file_paths.API_SIMILARITY_PATH, 'shape_info.yaml'), 'r')
        self.shape_dictionary = yaml.full_load(f)
        f.close()

    def get_shape_fix_sentence(self, api_sentence: str) -> str:
        result = '        x = x.reshape(-1, '

        # 获取api_name
        api_name = api_sentence.replace(' ', '')
        api_name = api_name.split('=')[1].split('(')[0]
        # api_name获取完成

        # 获取维度约束 和 格式约束
        ac_dims = self.get_acceptable_dims(api_name)
        ac_formats = self.get_acceptable_formats(api_name)
        # 维度约束 和 格式约束获取完成

        # 获取参数列表
        para_dict = {}
        para_list = api_sentence.replace(' ', '').split('(')[1].split(')')[0].split(',')
        if para_list[0] != '':
            for para in para_list:
                para_name = para.split('=')[0]
                para_val = para.split('=')[1]
                para_dict[para_name] = para_val
        # 参数列表获取完成

        # 1.无需变形的情况：
        if len(ac_dims) >= 4:
            if isinstance(ac_formats[0], str) and ac_formats[0] == 'any':
                return '\n'
        # ===

        # 2.选取可用维度 和 格式：
        index, dim = random.choice(list(enumerate(ac_dims)))
        format = ac_formats[index]
        # ===

        # 3.生成塑形语句
        if isinstance(format, str) and format == 'any':
            for i in range(dim):
                if i == 0:
                    continue
                else:
                    result = result + '8, '
            result = result[:-2]
            result = result + ')'
            result = result + '\n'
            return result
        elif isinstance(format, tuple):
            formatlist = list(format)
            for i in range(dim):
                if i == 0:
                    continue
                else:
                    if isinstance(formatlist[i], str):
                        if formatlist[i] in para_dict.keys():
                            result = result + para_dict[formatlist[i]] + ', '
                        else:
                            result = result + '8, '
                    else:
                        result = result + '8, '
            result = result[:-2]
            result = result + ')'
            result = result + '\n'
            return result
        else:
            return '\n'
        # ===

    # ==============================================
    # toolbox
    # ==============================================

    def get_shape_info_dictionary(self, api_name: str) -> dict:
        return self.shape_dictionary[api_name]

    def get_acceptable_dims(self, api_name: str) -> list:
        d = self.get_shape_info_dictionary(api_name)
        return d['acceptable_dims']

    def get_acceptable_formats(self, api_name: str) -> list:
        d = self.get_shape_info_dictionary(api_name)
        return d['acceptable_formats']


if __name__ == '__main__':
    api_sentence = '        self.conv1 = jittor.nn.Conv2d(in_channels = 3, out_channels = 16, kenel_size = 3)'
    sf = ShapeFixer()
    print(sf.get_shape_fix_sentence(api_sentence))
