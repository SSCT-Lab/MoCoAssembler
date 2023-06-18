import file_paths
import os
import yaml

path = os.path.join(file_paths.MAIN_PATH, '..', 'pytorch_', 'constraints', 'pytorch_original')
file_names = os.listdir(path)
layer_dictionary = {}
for file_name in file_names:
    layer_name = file_name.split('.')[2]
    layer_dictionary[layer_name] = 1
f = open(os.path.join(file_paths.API_SIMILARITY_PATH, 'layers_to_mutate.yaml'), 'w')
yaml.dump(layer_dictionary, f)