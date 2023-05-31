import os

CUR_PATH = os.path.dirname(__file__)

ROOT_PATH = os.path.dirname(CUR_PATH)

CONSTRAINTS_PATH = os.path.join(ROOT_PATH, "pytorch_", "constraints", "pytorch_modified")

DESCP_SOURCE_PATH = os.path.join(ROOT_PATH, "pytorch_", "constraints", "source_files")

MAPPING_FILE = os.path.join(ROOT_PATH, "pytorch_", "constraints", "category_mapping.yaml")

SIMPLE_MODEL_PATH = os.path.join(ROOT_PATH, "model", "pytorch_version", "simple_models")

COMPLEX_MODEL_PATH = os.path.join(ROOT_PATH, "model", "pytorch_version", "complex_models")

MUTATED_MODEL_PATH = os.path.join(ROOT_PATH, "result", "pytorch_version")

SIMILARITY_PATH = os.path.join(ROOT_PATH, "pytorch_", "data", "param")

DEF_SIMI_TARGET_PATH = os.path.join(ROOT_PATH, "pytorch_", "data", "function", "def_sim")

PARAM_SIMI_TARGET_PATH = os.path.join(ROOT_PATH, "pytorch_", "data", "function", "param_sim")

SIMI_TARGET_PATH = os.path.join(ROOT_PATH, "pytorch_", "data", "function", "sim")
