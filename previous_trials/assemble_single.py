import json
import os
import re


def get_apis_of_parent_model(parent_model):
    #获取run.py文件的目录路径
    cur_path=os.path.dirname(os.path.abspath(__file__))
    parent_model_path=os.path.join(cur_path+os.path.sep+"new_models",parent_model)
    save_parent_model_apis_path=os.path.join(cur_path+os.path.sep+"apis",parent_model[:-3]+"_apis.json")
    result = {}
    with open(parent_model_path, "r", encoding="utf-8") as f1,open(save_parent_model_apis_path, "w+", encoding="utf-8") as f2:
        #获取模型中所有可能的api
        flag=False#标记要改写的代码
        for line in f1:
            if "KitModel" in line:
                flag=True
            if "def" in line and "KitModel" not in line:
                flag=False
            if flag:
                if re.search(r'    .* +=', line) != None :
                    api_name, api_value = line.split('=', 1)
                    layer_key = "".join((api_name))
                    result.setdefault(layer_key, []).append(line)
        json_data=json.dumps(result)
        f2.write(json_data)


def remove_api_from_model(parent_model,generation,position,parent_model_name_length):
    code = []
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_model_path = os.path.join(cur_path + os.path.sep + "new_models", parent_model)
    #这里的[-3]不能改
    parent_model_layers_path = os.path.join(cur_path + os.path.sep + "apis", parent_model[:-3] + "_apis.json")
    with open(parent_model_path, "r", encoding="utf-8") as f1,open(parent_model_layers_path, 'r+') as f2 :
        content = f2.read()
        apis = json.loads(content)
        apis_length=len(list(apis.keys()))
        print("长度是：",apis_length)
        print("去掉的层是",position)
        #要去掉的层是
        rm_api_name=list(apis.keys())[position]
        print("rm_api:",rm_api_name)
        #去掉的层之前的层是
        pre_api_name=list(apis.keys())[position-1]#得到的是键
        print("pre_api",pre_api_name)
        #记录之前层的最后一个变量名，以修改代码
        #print("pre_api",pre_api)
        pre_api_name_real=pre_api_name.lstrip()
        pre_api_name_real=pre_api_name_real.strip()
        remd_api_name_real=rm_api_name.lstrip()
        remd_api_name_real=remd_api_name_real.strip()
        for line in f1:
            if line in apis.get(rm_api_name) and "flatten" not in rm_api_name:
               continue

            else:
                code.append(line)

    if "flatten" in rm_api_name:
        return True
    generation=str(generation)
    child_model_path = os.path.join(cur_path + os.path.sep + "new_models" + os.path.sep + parent_model[:parent_model_name_length] + "_" + generation+ ".py")
    with open(child_model_path,"w+", encoding="utf-8") as f1:
        for i in range(len(code)):
            if "    set_layer_weights" in code[i]:
               continue
            #更改后续代码

            if remd_api_name_real in code[i] :
                str1=code[i]
                str1=str1.replace(remd_api_name_real,pre_api_name_real)
                f1.write(str1)
            else:
                f1.write(code[i])


    return False

def get_models_list(modelRootPath):
    import os
    model_name_list=list()
    for i in os.listdir(modelRootPath):
        model_name_list.append("".join(i))
    return model_name_list

def get_parent_model_apis_length(parent_model):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_model_apis = os.path.join(cur_path + os.path.sep + "apis", parent_model[:-3] + "_apis.json")
    with open(parent_model_apis, 'r+') as f1:
        content = f1.read()
        content = json.loads(content)
        apis_length = len(list(content.keys()))
        return apis_length

def get_init_model_name_length(model_root_path):
    import os
    model_name_list = list()
    for i in os.listdir(model_root_path):
        model_name_list.append("".join(i))
    return len(model_name_list[0])-3

if __name__ == '__main__':
    flag=True
    listed = []
    root_path = './new_models'
    generation = 0
    parent_model_name_length = get_init_model_name_length(root_path)
    flatten_meet=False
    while(flag):
        print("listed:",listed)
        models=get_models_list(root_path)
        models.remove('__pycache__')
        model_list = list(set(models) ^ set(listed))
        print("model_list:", model_list)
        rm_apis_postion=4
        for parent_model in model_list:
            generation = generation + 1
            #print("母本是：",parent_model)
            get_apis_of_parent_model(parent_model)
            parent_model_apis_length = get_parent_model_apis_length(parent_model)
            if (parent_model_apis_length <=8 ):
                flag = False
                break
            listed.append(parent_model)
            flatten_meet=remove_api_from_model(parent_model, generation,rm_apis_postion,parent_model_name_length)
            if flatten_meet:
                break

        if flatten_meet:
            break

            # for i in range(len(rm_layer_position_list)):
            #     remove_layer_from_model(parent_model,generation,rm_layer_position_list[i])
