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
tf_func_file = Path.joinpath(tf_file, "data/function")

# tf函数定义相似度
tf_func_def_file = Path.joinpath(tf_func_file, "def_sim")

# tf函数参数列表相似度
tf_func_param_file = Path.joinpath(tf_func_file, "param_sim")

# tf函数相似度
tf_func_sim_file = Path.joinpath(tf_func_file, "sim")

# tf参数变异参数储存位置
tf_param_file = Path.joinpath(tf_file, "data/param")
