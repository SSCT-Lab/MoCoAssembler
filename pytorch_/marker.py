import os

ROOT_PATH = os.path.dirname(__file__)

CONSTRAINTS_PATH = os.path.join(ROOT_PATH, "constraints", "pytorch_modified")

DESCP_SOURCE_PATH = os.path.join(ROOT_PATH, "constraints", "source_files")

MAPPING_FILE = os.path.join(ROOT_PATH, "constraints", "category_mapping.yaml")

MUTATED_MODEL_PATH = os.path.join(ROOT_PATH, "mutated_models")

SIMPLE_MODEL_PATH = os.path.join(ROOT_PATH, "original_models", "simple_models")
