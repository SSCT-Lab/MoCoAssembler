import os

lst = os.listdir(os.path.join(os.getcwd(), "param"))
lst.sort()

with open(os.path.join(os.getcwd(), "api_list"), "w", encoding="utf8") as f:
    for item in lst:
        f.write(item[6:-5] + "\n")
    f.close()
