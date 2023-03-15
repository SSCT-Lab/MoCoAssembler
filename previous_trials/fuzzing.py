import os


def get_models_list(modelRootPath):
    import os
    model_name_list=list()
    for i in os.listdir(modelRootPath):
        model_name_list.append("".join(i))
    return model_name_list

import re

if __name__ == '__main__':
    model_list=get_models_list("vgg_models")
    #model_list.remove("__pycache__")
    print(model_list)
    with open("main.py", "r+", encoding="utf-8") as f1:
        content = f1.read()

        for i in range(2,len(model_list)):
            index=str(i)
            print(model_list[i])
            new_string="from new_models.new_"+index+" import KitMode"
            content=re.sub(r"from new_models.new_\d+ import KitMode",new_string,content)
            with open("new_run.py","w+",encoding="utf-8") as f2:
                f2.write(content)
            os.system("python ./"+"new_run.py")



