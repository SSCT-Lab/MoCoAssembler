import json
import os

def get_layers_of_parent_model(parent_model_content):
    parent_model_apis=[]
    #获取模型中所有可能的api
    flag_forward=False#标记要改写的代码
    flag_model=False
    api_def_list=[]
    for line in parent_model_content:
        if "KitModel" in line:
            flag_model=True
        if "forward" in line:
            flag_model=False
            flag_forward=True
        if "return" in line:
            flag_forward=False
        if flag_model:
            api_def_list.append(line)
        if flag_forward and 'forward' not in line:
                parent_model_apis.append(line)

    api_def_list=api_def_list[3:]
    #根据获取的apis得到模型的层layers，以便后面按照层删除
    result = {}  # empty dict
    #pre_api_name=None
    for line in parent_model_apis:
        left, right = line.split('=', 1)
        left="".join(left)
        right="".join(right)
        # 几个api形成一个共同的层，所以有相同前缀的api是一层,
        layer_key=left
        #var_name=layer_key.strip().lstrip()
        api_name_invocal=right[:right.find("(")].strip().lstrip()
        layer_dict={}
        #layer_dict['var_name']=var_name
        #layer_dict['api_name_invocal']=api_name_invocal
        for api_def in api_def_list:
            if api_name_invocal in api_def:
                layer_dict['api_def']=api_def
                break
            else:
                layer_dict['api_def']=None
        # layer_dict['var_value']=right
        layer_dict['line']=line
        #layer_dict['pre_api_name']=pre_api_name
        result.setdefault(layer_key, []).append(layer_dict)
        #pre_api_name=layer_key
    return result

def add_api(parent_model_content,layers,postion,init_layers=None):
    code = []
    index=postion
    add_api_name = list(init_layers.keys())[index]
    add_line=init_layers.get(add_api_name)[0]['line']
    add_api_def=init_layers.get(add_api_name)[0]['api_def']
    add_api_name_real = add_api_name.lstrip().strip()
    # 之前的层是
    pre_api_name = list(layers.keys())[index - 1]
    pre_line=layers.get(pre_api_name)[0]['line']
    pre_api_name_real = pre_api_name.lstrip().strip()
    #后层
    next_api_name = list(layers.keys())[index]
    next_code = layers.get(next_api_name)[0]['line']
    _, next_code_right = next_code.split("=", 1)
    next_code_right = "".join(next_code_right)
    next_code_right = next_code_right.replace(pre_api_name_real,add_api_name_real)
    next_new_code = next_api_name + "=" + next_code_right
    update_flag=False
    def_postion=1
    index=0
    for line in parent_model_content:
        if 'def forward' in line:
            def_postion=index
        if pre_line in line:
            code.append(line)
            code.append(add_line)
            if add_api_def is not None:
                code.insert(def_postion,add_api_def)
            update_flag=True
            continue
        elif update_flag:
            code.append(next_new_code)
            update_flag=False
        else:
            code.append(line)
        index=index+1
    return code

def save_all_layers(model_name,layers):
    model_name=model_name[10:-3]
    layers_path='model_layers' + os.path.sep + model_name+'.json'
    with open(layers_path,'w+',encoding="utf-8") as f1:
        json_data = json.dumps(layers)
        f1.write(json_data)

def get_header_tail(len):
    ls=[0,1]
    for i in range(len+1):
        if i<45:
            continue
        else:
            ls.append(i)
    return ls




