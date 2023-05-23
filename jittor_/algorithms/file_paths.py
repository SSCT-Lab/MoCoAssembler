import os
ALGORITHM_PATH = os.path.dirname(__file__)
MAIN_PATH = os.path.join(ALGORITHM_PATH, '..')
SIMPLE_MODEL_PATH = os.path.join(MAIN_PATH, 'seed_models', 'simple_models')
API_INFO_PATH = os.path.join(MAIN_PATH, 'jittor_api_info')
API_SIMILARITY_PATH = os.path.join(MAIN_PATH, 'jittor_api_similarity')
API_SIMILARITY_FILE_PATH = os.path.join(API_SIMILARITY_PATH, 'api_similarity.yaml')
LAYER_INFO_PATH = os.path.join(MAIN_PATH, 'jittor_layer_info')
LAYER_SIMILARITY_PATH = os.path.join(MAIN_PATH, 'jittor_layer_similarity')
LAYER_SIMILARITY_FILE_PATH = os.path.join(LAYER_SIMILARITY_PATH, 'layer_similarity.yaml')
NN_SOURCE_FILE_PATH = os.path.join(MAIN_PATH, 'nn_source_file')
MUTATED_MODEL_PATH = os.path.join(MAIN_PATH, 'mutated_model')


