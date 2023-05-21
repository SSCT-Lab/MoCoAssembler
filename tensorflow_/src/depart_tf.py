from config.paths import tf_model_file, tf_tmp_file
from config.keywords import INPUT_TENSOR, OUTPUT_TENSOR
from pathlib import Path


def depart(model_name):

    dir_org = tf_model_file
    dir_tmp = tf_tmp_file

    input_tensor = INPUT_TENSOR
    output_tensor = OUTPUT_TENSOR

    # 如果模版文件不存在，则创建新文件
    if not Path.exists(dir_tmp):
        Path.mkdir(dir_tmp)

    file_path_org = Path.joinpath(dir_org, (model_name + ".py"))
    file_path_tmp = Path.joinpath(dir_tmp, (model_name + "_tmp.py"))
    file_path_res = Path.joinpath(dir_tmp, (model_name + "_res.py"))

    file_tmp = Path.open(file_path_tmp, "w+", encoding="utf8")
    file_res = Path.open(file_path_res, "w+", encoding="utf8")

    with Path.open(file_path_org, "r", encoding="utf8") as file_org:

        for line in file_org:
            if line.find(input_tensor) >= 0:
                file_tmp.write("# " + model_name + " input layer" + "\n")
                file_tmp.write(line)
                break
            file_tmp.write(line)

        for line in file_org:
            if line.find(output_tensor) >= 0:
                file_tmp.write("\n")
                file_tmp.write("# " + model_name + " output layer" + "\n")
                file_tmp.write(line)
                break
            file_res.write(line)

        for line in file_org:
            file_tmp.write(line)

        file_org.close()
        file_tmp.close()


if __name__ == "__main__":
    depart("a_test")
