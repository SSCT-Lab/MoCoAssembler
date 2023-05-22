import os

CUR_PATH = os.path.dirname(__file__)

ROOT_PATH = os.path.dirname(CUR_PATH)

PT_ROOT_PATH = os.path.join(ROOT_PATH, "pytorch_")

CONSTRAINTS_PATH = os.path.join(PT_ROOT_PATH, "constraints", "pytorch_modified")

DESCP_SOURCE_PATH = os.path.join(PT_ROOT_PATH, "constraints", "source_files")

MAPPING_FILE = os.path.join(PT_ROOT_PATH, "constraints", "category_mapping.yaml")

MUTATED_MODEL_PATH = os.path.join(ROOT_PATH, "result", "pytorch_version")

SIMPLE_MODEL_PATH = os.path.join(ROOT_PATH, "model", "pytorch_version", "simple_models")

COMPLEX_MODEL_PATH = os.path.join(ROOT_PATH, "model", "pytorch_version", "complex_models")

if __name__ == "__main__":
    print(PT_ROOT_PATH)
