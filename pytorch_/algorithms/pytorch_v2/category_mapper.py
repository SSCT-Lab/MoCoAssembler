import os
import yaml
import marker


def map_category() -> None:
    constraint_path = os.path.join(marker.ROOT_PATH, "constraints", "pytorch_modified")
    category_list = os.listdir(constraint_path)

    content = {}

    for category in category_list:
        cur_path = os.path.join(constraint_path, category)
        file_list = os.listdir(cur_path)
        for file in file_list:
            if "torch.nn." in file:
                content[file[:-5]] = category

    print(content)

    with open(marker.MAPPING_FILE, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True)
        f.close()


def get_category(api: str) -> str:
    with open(marker.MAPPING_FILE, "r", encoding="utf-8") as f:
        content = yaml.full_load(f.read())
        f.close()

    if api not in content.keys():
        raise Exception("API " + api + " not included")

    return content[api]


if __name__ == '__main__':
    map_category()
    # print(get_category("torch.nn.ReLU"))
