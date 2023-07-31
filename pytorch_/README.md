# MoCo for PyTorch
### PYTORCH VERSION
2.0.0

### REQUIREMENTS
run MoCo_PyTorch need some libs, just pip them:
```
pip install pyyaml
pip install alive_progress
pip install openpyxl
```

### DIRECTORY
All files about MoCo_PyTorch is in this directory(pytorch_), but the datasets used for training is in the main directory.
There are several directory in pytorch_:

```pytorch_/algorithms``` -- algorithms .

```pytorch_/torch_layer_info``` -- layers info which is written in yaml files.

```pytorch_/torch_layer_similarity``` -- layers similarity info which is written in yaml files.

```pytorch_/seed_models``` -- seed model files.

And then, several directories will be created(auto) during running to contain running results.

### RUN
I. Fuzzing

Open your console under pytorch_/algorithms and type this to run MoCo FUZZING TEST:
```
python go_Fuzzing.py --MODEL 'lenet' --N 5
```
1. `MODEL` means the seed model used this time. Make sure that the model is in `'testnet'
'alexnet'
'lenet'
'ResNet18'
'ResNet50'
'InceptionV3'
'xception'
'mobilenet'
'squeezenet'
'vgg16'
'vgg19'
'densenet'
'BiLSTM'
'LSTM'
'GRU'`
. And if you want to run with all seed models in a loop, set para MODEL to `LOOP`.
2. `N` means times that MoCo will execute mutate operation in one generation. Set it to an int.

II. Boundary

Open your console under pytorch_/algorithms and type this for MoCo BOUNDARY TEST:
```
python go_Boundary.py --MODEL 'lenet'
```
`MODEL` means the seed model used this time. Make sure that the model is in the list above,
and if you want to run with all seed models in a loop, set para MODEL to `LOOP`.

### OUTPUT
The result will be written in several directories
(if not exists, these directories will be created while running):

```pytorch_/mutated_models``` -- models generated during running, but will be deleted every generation.

```pytorch_/saved_mutated_models``` -- error models will be saved here, they may show some potential bugs of pytorch.

```pytorch_/logs``` -- error logs will be recorded here.

```pytorch_/boundary_logs``` -- boundary test result.

```pytorch_/boundary_models``` -- models related to boundary test result.
