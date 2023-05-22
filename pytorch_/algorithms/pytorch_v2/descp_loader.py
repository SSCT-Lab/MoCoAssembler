import os
import yaml

import config.marker as marker
import category_mapper as mapper


def analyse_descp() -> None:
    print("Loading Starts...")

    category_list = os.listdir(marker.CONSTRAINTS_PATH)
    for category in category_list:
        cur_path = os.path.join(marker.CONSTRAINTS_PATH, category)
        file_list = os.listdir(cur_path)
        source_file = os.path.join(marker.DESCP_SOURCE_PATH, category)

        with open(source_file, "r", encoding="utf-8") as f:
            flag = False
            descp, cur_name = "", ""

            for line in f:
                if line.strip(' ') == '\n':
                    continue

                if flag:
                    if ' """' not in line:
                        descp += line.lstrip()
                        continue

                    cur_file = os.path.join(marker.CONSTRAINTS_PATH, category, cur_name + ".yaml")
                    with open(cur_file, "r", encoding="utf-8") as f1:
                        content = yaml.full_load(f1.read())
                        f1.close()

                    content["descp"] = descp

                    with open(cur_file, "w", encoding="utf-8") as f2:
                        yaml.dump(content, f2, allow_unicode=True)
                        f2.close()

                    flag = False
                    descp = ""

                # 这里要额外考虑init的情况
                keyword = "def " if category == "init" else "class "
                if keyword in line:
                    prefix = "torch.nn.init." if category == "init" else "torch.nn."
                    cur_name = prefix + line.split(" ")[1].split("(")[0]
                    file_name = cur_name + ".yaml"
                    if file_name in file_list:
                        flag = True

    print("Loading Ends...")


def get_constraint(api: str):
    content = {}
    try:
        category = mapper.get_category(api)
        constraint_file = os.path.join(marker.CONSTRAINTS_PATH, category, api + ".yaml")
        with open(constraint_file, "r", encoding="utf-8") as f:
            content = yaml.full_load(f.read())
            f.close()
    except Exception as err:
        print(err)

    return content


if __name__ == '__main__':
    analyse_descp()
    # print(get_constraint("torch.nn.Conv2d"))
