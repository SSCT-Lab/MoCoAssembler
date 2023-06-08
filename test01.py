# -*- coding: utf-8 -*-

"""
@Title   : 
@Time    : 2023/6/8 16:35
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""
from pathlib import Path

import yaml

path = Path('/Users/wuduo/Documents/导师组/MOCO/ModelAssembler/pytorch_/data/param')

lst = path.rglob('*.yaml')
s = set()
l = set()

for _ in lst:
    with open(_, "r", encoding="utf8") as file:
        data = yaml.load(file, yaml.Loader)
        constraints = data['constraints']
        for item in constraints.items():
            if 'dtype' not in item[1]:
                print(_.name + " " + item[0] + "没有dtype")
            else:
                if item[1]['dtype'] == 'int' and 'range' in item[1]:
                    if item[1]['range'] == None:

                        s.add(item[0])
                        l.add(_.name[:-5])
                        # print(_.name + " " + item[0])

print(s)

print(list(l))