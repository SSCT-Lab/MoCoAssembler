import os


def detection(keyword: str, filename: str):
    """
    用来过滤。。。
    在filename文件中过滤掉含有keyword的条目，
    要求一定要是这个格式：
    ==============================
            条         目
    ==============================
    """
    f = open(filename, "r", encoding="utf-8")
    info = f.read()
    f.close()
    info_list = info.split("==============================\n==============================\n")
    new_info_list = []
    for i in info_list:
        if keyword not in i:
            new_info_list.append(i)
        else:
            continue
    f = open("./过滤结果.txt", "w", encoding="utf-8")
    f.write("==============================\n==============================\n".join(new_info_list))
    f.close()
    return


def calculate_models_num(dir_name: str, calculate_boundary: bool) -> int:
    base_pth = os.path.join("../result", "result", dir_name)
    file_list = os.listdir(base_pth)
    file_list.remove("main_report.txt")
    base_num = len(file_list) - 1
    if not calculate_boundary:
        return base_num
    else:
        for file in file_list:
            boundary_pth = os.path.join(base_pth, file, "boundary_cases")
            if not os.path.exists(boundary_pth):
                continue
            f_list = os.listdir(boundary_pth)
            base_num += (len(f_list) - 1)
        return base_num
