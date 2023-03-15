import random
import re
import subprocess
import json


def mutate_value(api):
    new_api={}
    left,right=api.split("=",1)
    left="".join((left))
    right="".join(right)
    new_api.setdefault(left,[]).append(api)
    index=right.find("(")
    api_name=right[:index]
    #print(api_name)
    signature=right[index+1:-1]
    code=[]
    with open("temp.py", "r", encoding="utf-8") as f2:
        for line in f2:
            if "get_api_args()" in line:
                new_line=f"    get_api_args({signature})\n"
                code.append(new_line)
            else:
                code.append(line)
    with open("get_args.py","w+",encoding="utf-8") as f1:
        for i in range(len(code)):
            f1.write(code[i])

    subprocess.run(["python", "get_args.py"], shell=False, timeout=10)

    with open("args.json","r+",encoding="utf-8") as f3:
        content = f3.read()
        api_args = json.loads(content)

    keywords=dict(api_args['keyword'])
    args_list = get_args("torch.nn.Conv2d")
    # print(keywords)
    # print(args_list)
    for keyword in keywords.keys():
        if keyword in args_list and "in" not in keyword and "out" not in keyword:
            value=keywords[keyword]
            print(value)
            print(type(value))
            if keyword=='kernel_size':
                api_args['keyword'][keyword]=[5,4]

    #print(api_args)
    #print(new_api)

    args_str=""
    args=api_args['args']
    for i in range(len(args)):
      args_str+="%s,"%(args[i])
    #print("args_str",args_str)
    kwargs_str=""
    kwargs=dict(api_args['keyword'])
    for key in kwargs.keys():
       if isinstance(kwargs[key],str):
           kwargs_str += "%s='%s'," % (key, kwargs[key])
       else:
           kwargs_str += "%s=%s," % (key, kwargs[key])
    #print(kwargs_str)
    args_all_str=args_str+kwargs_str
    #print(args_all_str)

    new_api_str = f"{left} = {api_name}({args_all_str})\n"
    return new_api_str


def get_args(api_name):
    with open("torch_APIdef.txt", "r", encoding="utf-8") as f1:
        for line in f1:
            if api_name in line:
                api_def=line
                break
    start=len(api_name)
    signature=api_def[start+1:-2]
    args=signature.split(",")
    args_list=[]
    for arg in args:
        arg=arg.lstrip()
        arg=arg.strip()
        if "=" in arg:
            index=arg.find("=")
            arg=arg[:index]
        args_list.append(arg)

    return args_list



if __name__ == '__main__':
    api = "        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)"
    new_api_str=mutate_value(api)
    print(new_api_str)







