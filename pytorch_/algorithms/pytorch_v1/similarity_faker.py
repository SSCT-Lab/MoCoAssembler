import os
import random
import yaml

import marker

if __name__ == '__main__':
    constraint_path = os.path.join(marker.ROOT_PATH, "constraints", "pytorch_modified")
    category_list = os.listdir(constraint_path)

    for category in category_list:
        cur_path = os.path.join(constraint_path, category)
        lst_pre = os.listdir(cur_path)

        lst = []
        for item in lst_pre:
            if "torch.nn" in item:
                lst.append(item)

        content = {}
        for item1 in lst:
            temp = {}
            for item2 in lst:
                if item1 != item2:
                    item2_name = item2[:-5]
                    temp[item2_name] = random.random()
            item1_name = item1[:-5]
            content[item1_name] = temp

        print(content)

        with open(os.path.join(cur_path, "similarity.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(content, f, allow_unicode=True)
