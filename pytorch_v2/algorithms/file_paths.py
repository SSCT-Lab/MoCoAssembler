import os
ALGORITHM_PATH = os.path.dirname(__file__)
MAIN_PATH = os.path.join(ALGORITHM_PATH, '..')
MODEL_PATH = os.path.join(MAIN_PATH, 'seed_models')

LAYER_INFO_PATH = os.path.join(MAIN_PATH, 'torch_layer_info')

LAYER_SIMILARITY_PATH = os.path.join(MAIN_PATH, 'torch_layer_similarity')

MUTATED_MODEL_PATH = os.path.join(MAIN_PATH, 'mutated_model')

SAVED_MODEL_PATH = os.path.join(MAIN_PATH, 'saved_mutated_models')

LOG_PATH = os.path.join(MAIN_PATH, 'logs')
