import os


def mutate_on_api(file_path: str):
    with open(file_path, "r", encoding="utf8") as f:

        f.close()
    return


def mutate_on_params(file_path: str):
    with open(file_path, "r", encoding="utf8") as f:

        f.close()
    return


if __name__ == '__main__':
    path = os.getcwd()
    model_dir = path + "/../model/pytorch_version/simple_models/"
    file_list = os.listdir(model_dir)

    for net in file_list:
        mutate_on_api(model_dir + net)
