from pathlib import Path

# 根目录
CUR_ROOT = Path.cwd()

# tensorflow
TF_PATH = CUR_ROOT.parent

# tf模型文件路径
TF_MODEL_PATH = TF_PATH.parent / "model/tensorflow_version"

# tf模型拆分后结果存储的文件路径
TF_RES_PATH = TF_PATH / "result"

# tf异常日志存储
TF_LOG_PATH = TF_PATH / "log"

# tf变异函数列表(json格式)
TF_FUNC_PATH = TF_PATH / "data/function"

# tf参数变异参数储存位置
TF_PARAM_PATH = TF_PATH / "data/param"

# tf函数定义相似度
TF_FUNC_DEF_PATH = TF_FUNC_PATH / "def_sim"

# tf函数参数列表相似度
TF_FUNC_PARAM_PATH = TF_FUNC_PATH / "param_sim"

# tf函数相似度
TF_FUNC_SIM_PATH = TF_FUNC_PATH / "sim"