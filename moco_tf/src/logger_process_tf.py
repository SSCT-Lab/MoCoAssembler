import csv
import re
import sys
import traceback
from pathlib import Path
sys.path.append(Path.cwd().parent.parent.__str__())

from moco_tf.config.paths import LOG_PATH
from moco_tf.config.models import models


def log_process(model_name):
    log_file_path = LOG_PATH
    model_file = log_file_path / model_name
    if is_dir_empty(model_file):
        print(model_file + " is empty.")

    error_type = ""
    log_path = ""
    error_path = ""
    error_message = ""
    error_code = ""
    error_line_number = -1

    data = []

    epoch_list = [entry for entry in model_file.iterdir() if entry.is_dir()]

    for _ in epoch_list:
        if is_dir_empty(_):
            print(str(_) + " is empty")
        else:
            log_list = [entry for entry in _.iterdir() if entry.is_file()]
            for __ in log_list:
                with Path.open(__, "r", encoding="utf8") as error_file:

                    log_path = str(__).strip()

                    trackback_list = error_file.readlines()
                    message = []
                    for i in range(len(trackback_list)):
                        line = trackback_list[i]
                        if line.find("in " + model_name) >= 0:
                            error_code = trackback_list[i + 1].strip()
                            error = re.findall(r'"([^"]*)"|(\d+)', line)
                            for match in error:
                                if match[0]:
                                    error_path = str(match[0]).strip()
                                elif match[1]:
                                    error_line_number = int(match[1])
                        if line.find("Error:") >= 0:
                            message.append(line.strip())

                    error_message = " ".join(message)
                    error_type = error_message.split(":")[0]

                with Path.open(__, "r", encoding="utf8") as error_file:
                    message = []
                    for line in error_file:
                        if line.find("Error") >= 0:
                            message.append(line.strip())
                            break

                    for line in error_file:
                        if line == "\n":
                            pass
                        message.append(line.strip())

                    error_message = " ".join(message)

                try:
                    error_path = error_path[error_path.index("/result"):]
                    log_path = log_path[log_path.index("/log"):]
                except:
                    continue

                dict = {
                    "error_type": error_type.strip(),
                    "log_path": log_path,
                    "error_path": error_path,
                    "error_message": error_message.strip(),
                    "error_code": error_code.strip(),
                    "error_line": error_line_number.__str__()
                }
                data.append(dict)

    if len(data) == 0:
        pass
    else:
        fieldnames = data[0].keys()
        with Path.open(model_file / "log.csv", "w", encoding="utf8") as log_file_sum:
            writer = csv.DictWriter(log_file_sum, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        # 对分练过的csv文件排序
        sort_csv(model_file)

        # 二次排序
        # sort_sort_csv(model_file)

        # 对排序去重
        remove_duplicates(model_file)

        # 过滤
        filter(model_file)


def sort_csv(_):
    rows = csv_loads(_ / "log.csv")

    sorted_rows = sorted(rows, key=lambda x: (x["error_type"], x["error_message"], x["error_code"]))

    csv_dumps(_ / "log_sorted.csv", sorted_rows)


def sort_sort_csv(_):
    rows = csv_loads(_ / "log_sorted.csv")
    sort_sort_rows = []
    tmp_rows = []
    type = ""
    for row in rows:
        if row["error_type"] == type:
            tmp_rows.append(row)
        else:
            tmp_rows.append(row)
            if len(tmp_rows) == 0:
                pass
            else:
                tmp_rows = sorted(tmp_rows, key=lambda x: x["error_message"])
                sort_sort_rows.extend(tmp_rows)
                tmp_rows = []
        type = row["error_type"]

    tmp_rows = sorted(tmp_rows, key=lambda x: x["error_message"])
    sort_sort_rows.extend(tmp_rows)

    csv_dumps(_ / "log_sorted_.csv", sort_sort_rows)


def remove_duplicates(_):
    rows = csv_loads(_ / "log_sorted.csv")

    unique_rows = []
    last_row3, last_row4 = "", ""
    for row in rows:
        if row["error_message"] == last_row3:
            continue
        else:
            unique_rows.append(row)

        last_row3, last_row4 = row["error_message"], row["error_code"]

    csv_dumps(_ / "log_removal_dup.csv", unique_rows)


def filter(_):
    rows = csv_loads(_ / "log_removal_dup.csv")

    filter_rows = []
    for row in rows:
        if len(re.findall(r".*?expected.*?found.*?", row["error_message"])) > 0:
            # 过滤掉形状不同的
            continue
        elif len(re.findall(r".*?Exception encountered when calling layer.*?Negative dimension",
                            row["error_message"])) > 0:
            # 过滤掉由于padding值不对而产生的问题
            continue
        elif len(re.findall(r".*?Argument `cropping` must be greater than the input shape.*?",
                            row["error_message"])) > 0:
            # 过滤cropping错误
            continue
        elif len(re.findall(r".*?Attention layer must be called on a list of inputs.*?",
                            row["error_message"])) > 0:
            # 过滤AdditiveAttention输入错误
            continue
        elif len(re.findall(r".*?layer should be called on a list of.*?", row["error_message"])) > 0:
            # 过滤输入错误
            continue
        elif len(re.findall(r".*?One of the dimensions in the output is <= 0.*?", row["error_message"])) > 0:
            # 过滤掉由于padding值不对而产生的问题
            continue
        elif len(re.findall(r".*?missing 1 required positional argument: 'states'.*?",
                            row["error_message"])) > 0:
            # 过滤rnn网络的states问题
            continue
        elif len(re.findall(r".*?`dim` must be in the range.*?", row["error_message"])) > 0:
            # 过滤dim错误
            continue
        elif len(re.findall(r".*?rank.*?", row["error_message"])) > 0:
            # 过滤rank错误
            continue
        elif len(re.findall(r".*?Strides must be greater than output padding.*?", row["error_message"])) > 0:
            # 过滤Strides must be greater than output padding
            continue
        elif len(re.findall(r".*?`strides > 1` not supported in conjunction with `dilation_rate > 1`.*?", row["error_message"])) > 0:
            # 过滤`strides > 1` not supported in conjunction with `dilation_rate > 1`
            continue
        elif len(re.findall(r".*?'images' must have either.*? or .*? dimensions.*?",
                                row["error_message"])) > 0:
            # 过滤lambda错误
            continue
        elif len(re.findall(r".*?`interpolation` argument should be one of.*?",
                                row["error_message"])) > 0:
            # 过滤interpolation错误
            continue
        elif len(re.findall(r".*?list index out of range.*?",
                                row["error_message"])) > 0:
            # 过滤列表超出错误
            continue

        elif len(re.findall(r".*?function=None.*?", row["error_code"])) > 0:
            # 过滤Lambda函数错误
            continue
        elif row["error_type"] == "tensorflow.python.framework.errors_impl.ResourceExhaustedError":
            # 过滤掉OOM
            continue
        else:
            filter_rows.append(row)

    csv_dumps(_ / "log_filter.csv", filter_rows)


def csv_loads(path):
    with Path.open(path, "r", encoding="utf8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    return rows


def csv_dumps(path, rows):
    if len(rows) > 0:
        fieldnames = rows[0].keys()
        with Path.open(path, "w", encoding="utf8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    else:
        print(str(path) + " should be empty, please make sure that file is not empty.")


def is_dir_empty(directory_path):
    directory = Path(directory_path)
    if directory.is_dir():
        return len(list(directory.iterdir())) == 0
    else:
        raise ValueError(f"{directory_path} is not a directory.")


if __name__ == "__main__":
    for model in models:
        try:
            log_process(model)
            print("{} success!\n".format(model))
        except Exception as e:
            print("{} wrong!\n".format(model) + traceback.format_exc())
