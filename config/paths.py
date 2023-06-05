from pathlib import Path

# 根目录
CUR_ROOT = Path.cwd()

# tensorflow
PATH = CUR_ROOT.parent

# tf模型文件路径
TF_MODEL_PATH = PATH.parent / "model/tensorflow_version"

# PYTORCH模型文件路径
PT_MODEL_PATH = PATH.parent / "model/pytorch_version"

# tf模型拆分后结果存储的文件路径
RES_PATH = PATH / "result"

# tf异常日志存储
LOG_PATH = PATH / "log"

# tf变异函数列表(json格式)
FUNC_PATH = PATH / "data/function"

# tf参数变异参数储存位置
PARAM_PATH = PATH / "data/param"

# tf函数定义相似度
FUNC_DEF_PATH = FUNC_PATH / "def_sim"

# tf函数参数列表相似度
FUNC_PARAM_PATH = FUNC_PATH / "param_sim"

# tf函数相似度
FUNC_SIM_PATH = FUNC_PATH / "sim"

