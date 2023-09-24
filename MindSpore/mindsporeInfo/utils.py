class Info:
    BASE_TYPE = ['int', 'str', 'float', 'bool']
    BASE_STRUCTURE = ['list', 'tuple', 'dict', 'single']

    def __init__(self, api, descp, param_list, optional, required):
        self.dict = {}
        self.api = api
        self.descp = descp
        self.param_list = param_list
        self.optional = optional
        self.required =required

    def print(self):
        print("api = {}".format(self.api))
        print("descp = {}".format(self.descp))
        print("param_list = {}".format(self.param_list))
        print("optional = {}".format(self.optional))
        print("required = {}".format(self.required))

    def dict_info(self) -> dict:
        constraints = {}
        for param in self.param_list:
            constraints[param.name] = self.descp_process(param.descp, param.default, param.dtype)

        inputs = dict([
            ('optional', self.optional),
            ('required', self.required)
        ])

        self.dict = dict([
            ('api', self.api),
            ('descp', self.descp),
            ('constraints', constraints),
            ('inputs', inputs)
        ])

        return self.dict

    def descp_process(self, descp, default, dtype) -> dict:
        param_dict = dict([
            ('descp', descp),
            ('default', default),
            ('dtype', dtype)
        ])

        tmp_dtype = dtype
        tmp_dtype = tmp_dtype.lower() if 'optional' not in tmp_dtype else tmp_dtype.lower().replace(', optional', '')
        # print(tmp_dtype)

        if 'union' in tmp_dtype:
            tmp_dtype = tmp_dtype[6:-1]

        tmp_dtype = [_.strip() for _ in tmp_dtype.split(',')]
        final_dtype = set()
        structure = set()
        enum = set()

        is_Union = False
        for type in tmp_dtype:
            if type in [self.BASE_TYPE[0], self.BASE_TYPE[2]]:  # int, float
                final_dtype.add(type)
                structure.add(self.BASE_STRUCTURE[3])
            elif type in [self.BASE_TYPE[1], self.BASE_TYPE[3]]:  # str, bool
                final_dtype.add(type)
            elif type in self.BASE_STRUCTURE:  # list, tuple, dict
                structure.add(type)
            elif type == 'none':
                final_dtype.add(self.BASE_TYPE[1])
                enum.add('None')

            elif self.BASE_STRUCTURE[0] in type:  # list[]
                undetermined_type = type[5:-1]
                structure.add(self.BASE_STRUCTURE[0])
                if undetermined_type in self.BASE_TYPE:
                    final_dtype.add(undetermined_type)
                else:
                    is_Union = True
                    final_dtype.add(dtype.lower())
            elif self.BASE_STRUCTURE[1] in type:  # tuple[]
                undetermined_type = type[6:-1]
                structure.add(self.BASE_STRUCTURE[1])
                if undetermined_type in self.BASE_TYPE:
                    final_dtype.add(undetermined_type)
                else:
                    is_Union = True
                    final_dtype.add(dtype.lower())
            else:
                # 表示有无法自动化处理的类型，使用初始化dtype
                is_Union = True
                final_dtype.add(dtype.lower())

        param_dict['dtype'] = list(final_dtype) if final_dtype else None
        for _ in final_dtype:
            if self.BASE_TYPE[0] == _:  # int
                param_dict['structure'] = list(structure) if structure else None
                param_dict['shape'] = None
                param_dict['range'] = None
            elif self.BASE_TYPE[1] == _:  # str:
                param_dict['enum'] = list(enum) if enum else None
            elif self.BASE_TYPE[2] == _:  # float
                param_dict['structure'] = list(structure) if structure else None
                param_dict['shape'] = None
            elif self.BASE_TYPE[3] == _:  # bool
                pass

        if structure:
            param_dict['structure'] = list(structure)

        # print(param_dict)
        return param_dict


class Param:
    def __init__(self, name, descp, default, dtype):
        self.name = name
        self.descp = descp.replace('\\', '').replace('\\u', '')
        self.default = default
        self.dtype = dtype

    def print(self):
        print("name = {}".format(self.name))
        print("descp = {}".format(self.descp))
        print("default = {}".format(self.default))
        print("dtype = {}".format(self.dtype))
