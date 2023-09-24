# -*- coding: utf-8 -*-

"""
@Title   : 
@Time    : 2023/9/19 16:36
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""

dtype_list = [
    'Union[tuple[int], list[int]]',
    'Union[int, None]',
    'Union[int, tuple[int]]',
    'Union[dict, None]',
    'Union[Tensor, float]',
    'Union[float, int, Tensor, Iterable, LearningRateSchedule]',
    'Union[float, list, Tensor]',
    'Union[float, int, Cell]',
    'Union[list[int], tuple[int]]',
    'Union[int, tuple[int]], optional',
    'Union[float, int, None]',
    'Union[str, Cell, Primitive, None]',
    'Union[int, float, Tensor], optional',
    'Union(int, tuple[int])',
    'Union(tuple[int], list[int])',
    'Union[None, tuple]',
    'Union[int, float], optional',
    'Union[float, int]',
    'Union[list, tuple]',
    'Union[int, float, Tensor]',
    'Union[int, float]',
    'Union[float, tuple[float]], optional',
    'Union[list(Parameter), list(dict)]',
    'Union[int, tuple]',
    'Union[str, None]',
    'Union[Cell, Primitive]',
    'Union[Tensor, Cell]',
    'Union[list[Parameter], list[dict]]',
    'Union[float, None]',
    'Union[Tensor, str, Initializer, numbers.Number]',
    'Union[float, Tensor]',
    'Union[Tensor, None]',
    'Union[str, callable, Cell]',
    'Union(int, tuple[int], list[int]), optional',
    'Union[Tensor, int, float]',
    'Union[list[float], tuple[float]]',
    'Union[str, Cell]',
    'Union[Cell]',
    'Union(int, tuple[int], list[int])',
    'int, optional',
    'list',
    'string',
    'float, optional',
    'int',
    'tuple[Tensor]',
    'mindspore.nn.Optimizer',
    'tuple',
    'mindspore.nn.optim_ex.Optimizer',
    'list[mindspore.dtype]',
    'dict',
    'Cell, optional',
    'list[tuple[int]]',
    'function',
    'FuncGraph',
    'object',
    'Parameter',
    'str',
    'dict, optional',
    'bool, optional',
    'str, optional',
    'float',
    'ParameterTuple',
    'Cell',
    'bool',
    'int, None',
    'numbers.Number',
    'Tensor',
    'float, int',
    'int, float',
    'Tensor, optional',
    'Function',
    'tuple[float, float]',
    'Tuple[float, float], optional',
    'mindspore.dtype',
]
BASE_TYPE = ['int', 'str', 'float', 'bool']
BASE_STRUCTURE = ['list', 'tuple', 'dict', 'single']

for dtype in dtype_list:
    tmp_dtype = dtype
    tmp_dtype = tmp_dtype.lower() if 'optional' not in tmp_dtype else tmp_dtype.lower().replace(', optional', '')
    print(tmp_dtype)

    param_dict = dict([('dtype', tmp_dtype)])
    if 'union' in tmp_dtype:
        tmp_dtype = tmp_dtype[6:-1]

    tmp_dtype = [_.strip() for _ in tmp_dtype.split(',')]
    final_dtype = set()
    structure = set()
    enum = set()

    is_Union = False
    for type in tmp_dtype:
        if type in [BASE_TYPE[0], BASE_TYPE[2]]:  # int, float
            final_dtype.add(type)
            structure.add(BASE_STRUCTURE[3])
        elif type in [BASE_TYPE[1], BASE_TYPE[3]]:  # str, bool
            final_dtype.add(type)
        elif type in BASE_STRUCTURE:  # list, tuple, dict
            structure.add(type)
        elif type == 'none':
            final_dtype.add(BASE_TYPE[1])
            enum.add('none')

        elif BASE_STRUCTURE[0] in type:  # list[]
            undetermined_type = type[5:-1]
            structure.add(BASE_STRUCTURE[0])
            if undetermined_type in BASE_TYPE:
                final_dtype.add(undetermined_type)
            else:
                is_Union = True
                final_dtype.add(dtype.lower())
        elif BASE_STRUCTURE[1] in type:  # tuple[]
            undetermined_type = type[6:-1]
            structure.add(BASE_STRUCTURE[1])
            if undetermined_type in BASE_TYPE:
                final_dtype.add(undetermined_type)
            else:
                is_Union = True
                final_dtype.add(dtype.lower())
        else:
            # 表示有无法自动化处理的类型，使用初始化dtype
            is_Union = True
            final_dtype.add(dtype.lower())

    param_dict['dtype'] = final_dtype if final_dtype else None
    for _ in final_dtype:
        if BASE_TYPE[0] == _:  # int
            param_dict['structure'] = structure if structure else None
            param_dict['shape'] = None
            param_dict['range'] = None
        elif BASE_TYPE[1] == _:  # str:
            param_dict['enum'] = enum if enum else None
        elif BASE_TYPE[2] == _:  # float
            param_dict['structure'] = structure if structure else None
            param_dict['shape'] = None
            param_dict['range'] = None
        elif BASE_TYPE[3] == _:  # bool
            pass

    if structure:
        param_dict['structure'] = structure

    print(param_dict)
    print('\n')
