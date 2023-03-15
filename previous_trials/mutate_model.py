import json
import os


def getApiToMutate(filename,position):
    with open(filename,"r+",encoding="utf-8") as f1:
        content = f1.read()
        layers = json.loads(content)
        layers=dict(layers)
        key=list(layers.keys())[position]
        layer=layers.get(key)[0]
        api_line=layer['line']
        api_def=layer['api_def']
        pre_api_name=list(layers.keys())[position-1]
        return api_line,api_def,pre_api_name


def generate_model_mutation(seed_model_file,layers_path,api_position):
    api_line,api_def,pre_api_name=getApiToMutate(layers_path,api_position)
    from mutate_api_value import mutate_api
    if api_def is not None:
        print("old_api: ",api_def)
        new_api=mutate_api(api_def,pre_api_name=pre_api_name)
    else:
        print("old_api: ",api_line)
        new_api=mutate_api(api_line,pre_api_name=pre_api_name)

    print("new_api: ",new_api)
    print("-"*10)
    #保存新的模型
    new_model_name=seed_model_file[11:-3]+'_mutate.py'
    new_model_path=os.path.join('model_mutation' + os.path.sep , new_model_name)
    with open(seed_model_file,"r+",encoding="utf-8") as f1,open(new_model_path,"w+",encoding="utf-8")as f2:
        contend=f1.read()
        if api_def is not None:
            new_py=contend.replace(api_def,new_api)
        else:
            print(api_line in contend)
            new_py=contend.replace(api_line,new_api)
        f2.write(new_py)


