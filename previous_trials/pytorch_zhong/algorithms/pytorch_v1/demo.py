import os

import yaml

import marker


def demo():
    lst1 = ["activation", "batchnorm", "conv", "dropout", "linear", "pooling", "rnn", "sparse"]
    for category in lst1:
        st = []
        cur_dir = os.path.join(marker.CONSTRAINTS_PATH, category)
        lst2 = os.listdir(cur_dir)
        for name in lst2:
            if "torch.nn" not in name:
                continue
            file = os.path.join(cur_dir, name)
            with open(file, "r", encoding="utf-8") as f:
                content = yaml.full_load(f.read())
                f.close()
            for item in content["constraints"]:
                s0 = str(content["constraints"][item]["default"])
                if s0 == 'None':
                    print(file, item)
                st.append(s0)
        print(category)
        st = list(set(st))
        for s in st:
            print(s)
        print("\n")


if __name__ == "__main__":
    # ran = "[asdw123dq 3, asddasda]"
    # match1 = re.match(r'\[(.*),\s*(.*)\]', ran)
    # match2 = re.match(r'\[(.*),\s*(.*)\)', ran)
    # if match1:
    #     low, high = match1.group(1), match1.group(2)
    # elif match2:
    #     low, high = match2.group(1), match2.group(2)
    # else:
    #     low, high = 0, 64
    #
    # print(low, high)
    # current_time = datetime.datetime.now()
    # formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    #
    # print("Current time:", formatted_time)
    print(int("1112") < int("2"))
