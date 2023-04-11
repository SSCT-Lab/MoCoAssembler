from pathlib import Path

# 根目录
root = Path.cwd()

# tensorflow
tf_file = Path(root).parent

# tf模型文件路径
tf_model_file = Path.joinpath(tf_file, "model")

# tf模型模版文件路径
tf_tmp_file = Path.joinpath(tf_file, "template")

# tf模型变异文件路径
tf_mut_file = Path.joinpath(tf_file, "mutate")

# tf变异函数列表(json格式)
tf_func_file = Path.joinpath(tf_file, "data/function/json")

# tf参数变异参数储存位置
tf_param_file = Path.joinpath(tf_file, "data/param")

