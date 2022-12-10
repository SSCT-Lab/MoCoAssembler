import json
import os
import re

def get_layers_of_parent_model(parent_model):
    #获取run.py文件的目录路径
    cur_path=os.path.dirname(os.path.abspath(__file__))
    parent_model_path=os.path.join(cur_path+os.path.sep+"new_models",parent_model)
    save_parent_model_apis_path=os.path.join(cur_path+os.path.sep+"apis_to_layers",parent_model[:-3]+"_apis.txt")
    with open(parent_model_path, "r", encoding="utf-8") as f1,open(save_parent_model_apis_path, "w+", encoding="utf-8") as f2:
        #获取模型中所有可能的api
        flag=False#标记要改写的代码
        for line in f1:
            if "KitModel" in line:
                flag=True
            if "def" in line and "KitModel" not in line:
                flag=False
            if flag:
                if re.search(r'    \w+.*=', line) != None :
                    f2.write(line)

    #根据获取的apis得到模型的层layers，以便后面按照层删除
    result = {}  # empty dict
    with open(save_parent_model_apis_path, "r", encoding="utf-8") as f1:
        for api_line in f1:
            api_name, api_value = api_line.split('=', 1)
            #几个api形成一个共同的层，所以有相同前缀的api是一层,
            layer_key = re.findall(r"    .*_\d", api_name)#层的键
            #print(layer_key)
            layer_key="".join((layer_key))
            result.setdefault(layer_key, []).append(api_line)


    #将result存起来
    parent_model_layers=os.path.join(cur_path+os.path.sep+"apis_to_layers",parent_model[:-3]+"_layers.json")
    with open(parent_model_layers,"w+",encoding="utf-8") as f1:
        json_data = json.dumps(result)
        f1.write(json_data)



# def add_API_to_model(file):
#
#     code=[]
#     with open(file, "r", encoding="utf-8") as f1:
#         for line in f1:
#             code.append(line)
#             if re.search("weights_dict = load_weights_from_file",line)!=None:
#                 with open("apis.json", 'r+') as f2:
#                     content = f2.read()
#                     content = json.loads(content)
#                     # if content[]
#                     input_layers=content.get(list(content.keys())[0])
#                     for i in range(len(input_layers)):
#                         code.append(input_layers[i])
#                     output_layers=content.get(list(content.keys())[-2])
#                     for i in range(len(output_layers)):
#                         code.append(output_layers[i])
#                     last=content.get(list(content.keys())[-1])
#                     for i in range(len(last)):
#                         code.append(last[i])
#     for i in range(len(code)):
#         with open("new_model.py", "a", encoding="utf-8") as f1:
#            if "set_layer_weights(model, weights_dict)" in code[i]:
#               continue
#            f1.write(code[i])

def remove_layer_from_model(parent_model,position):
    code = []
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_model_path = os.path.join(cur_path + os.path.sep + "new_models", parent_model)
    parent_model_layers_path = os.path.join(cur_path + os.path.sep + "apis_to_layers", parent_model[:-3] + "_layers.json")
    with open(parent_model_path, "r", encoding="utf-8") as f1,open(parent_model_layers_path, 'r+') as f2 :
        content = f2.read()
        layers = json.loads(content)
        layers_length=len(list(layers.keys()))
        print("长度是：",layers_length)
        print("去掉的层是",position)
        #要去掉的层是
        rm_layer=list(layers.keys())[position]
        print("rm_layer:",rm_layer)
        #去掉的层之前的层是
        pre_layer=list(layers.keys())[position-1]#得到的是键
        print("prelayer",pre_layer)
        #记录之前层的最后一个变量名，以修改代码
        pre_layer_last_apis=layers.get(pre_layer)#得到之前层的值
        print("pre_layer_last_apis",pre_layer_last_apis)
        pre_layer_last_api=pre_layer_last_apis[-1]
        pre_layer_last_api="".join(pre_layer_last_api)
        pre_api_name,pre_api_value=pre_layer_last_api.split("=",1)
        pre_api_name = pre_api_name.lstrip()
        print(pre_api_name)
        rmed_layer_last_api = layers.get(rm_layer)[-1]
        print(rmed_layer_last_api)
        rmed_layer_last_api_name, rmed_layer_last_api_value = rmed_layer_last_api.split("=", 1)
        rmed_layer_last_api_name="".join(rmed_layer_last_api_name)
        rmed_layer_last_api_name=rmed_layer_last_api_name.lstrip()
        rmed_layer_last_api_name=rmed_layer_last_api_name.strip()
        print("rmed_layer_last_api_name:",rmed_layer_last_api_name)
        #if "flatten" in rm_layer:
        for line in f1:
            if line in layers.get(rm_layer) and "flatten" not in rm_layer:
               continue
            else:
                code.append(line)
    cishu=str(position)
    child_model_path = os.path.join(cur_path + os.path.sep + "new_models"+os.path.sep+parent_model[:-3]+"_"+cishu+".py")
    with open(child_model_path,"w+", encoding="utf-8") as f1:
        for i in range(len(code)):
            if "    set_layer_weights" in code[i]:
               continue
            #更改后续代码
            if rmed_layer_last_api_name in code[i]:
                str1=code[i]
                str1=str1.replace(rmed_layer_last_api_name,pre_api_name)
                f1.write(str1)
            else:
                f1.write(code[i])

def get_models_list(modelRootPath):
    import os
    model_name_list=list()
    for i in os.listdir(modelRootPath):
        model_name_list.append("".join(i))
    return model_name_list

def get_parent_model_layers_length(parent_model):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_model_layers = os.path.join(cur_path + os.path.sep + "apis_to_layers", parent_model[:-3] + "_layers.json")
    with open(parent_model_layers, 'r+') as f1:
        content = f1.read()
        content = json.loads(content)
        layers_length = len(list(content.keys()))
        return layers_length

if __name__ == '__main__':
    root_path='./new_models'
    model_list=get_models_list(root_path)
    model_list.remove('__pycache__')
    for parent_model in model_list:
        print("母本是：",parent_model)
        get_layers_of_parent_model(parent_model)
        parent_model_layers_length = get_parent_model_layers_length(parent_model)
        rm_layer_position_list=range(2,parent_model_layers_length-1)
        for i in range(len(rm_layer_position_list)):
            remove_layer_from_model(parent_model,rm_layer_position_list[i])
        break
