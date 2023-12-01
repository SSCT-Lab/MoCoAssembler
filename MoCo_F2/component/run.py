import time
import traceback
import CaseGenerator as cC
import os


def calculate_models_num(dir_name: str, calculate_boundary: bool) -> int:
    base_pth = os.path.join("../result", dir_name)
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


def go(net_name):
    time.sleep(1.0)
    cC.concrete.new_experiment()
    start_time = time.time()
    try:
        cC.goFuzzing(net_name)
        cC.generateBoundary(cC.concrete.get_experiment_id())
        end_time = time.time()
        error_report = ""
    except Exception:
        error_report = traceback.format_exc()
        end_time = time.time()
    report = "total time cost: " + str(end_time - start_time) + "\n" + str(time.time()) + "\n"
    report += error_report + "\n"
    fuzzing_num = calculate_models_num(cC.concrete.get_experiment_id(), False)
    boundary_num = calculate_models_num(cC.concrete.get_experiment_id(), True) - fuzzing_num
    report += "Fuzzing cases: " + str(fuzzing_num) + "\n"
    report += "Boundary cases: " + str(boundary_num) + "\n"
    report += "Average time cost of every fuzzing case: " + str((end_time - start_time) / fuzzing_num) + "\n"
    report += cC.concrete.get_experiment_id() + " -> " + net_name + "\n"
    f = open("../report/report_test_" + net_name + ".txt", "w", encoding="utf-8")
    f.write(report)
    f.close()


if __name__ == "__main__":
    # model_name = ["LeNet", "alexnet", "googlenet", "mobilenet", "squeezenet", "vgg16", "vgg19", "resnet18",
    #               "pointnet", "lstm"]
    # for model in model_name:
    #     go(model)
    go("pointnet")
