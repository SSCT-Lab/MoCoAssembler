# MoCo for Jittor
### Jittor version
1.3.7.16

### REQUIREMENTS
Running MoCo_Jittor need some libs, just pip them:
```
pip install pyyaml
pip install alive_progress
pip install openpyxl
```

### DIRECTORY
All files about MoCo_Jittor is in this directory(jittor_), but the datasets used for training is in the main directory.
There are several directory in jittor_:

```jittor_/algorithms``` -- algorithms .

```jittor_/jittor_layer_info``` -- layers info which is written in yaml files.

```jittor_/jittor_layer_similarity``` -- layers similarity info which is written in yaml files.

```jittor_/seed_models``` -- seed model files.

And then, several directories will be created(auto) during running to contain running results.

### RUN
I. Fuzzing

Open your console under jittor_/algorithms and type this for MoCo FUZZING TEST:
```
python go_Fuzzing.py --MODEL 'lenet' --N 5 --TSF 0
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
'GRU'
'googlenet'`
. And if you want to run with all seed models in a loop, set para MODEL to `LOOP`.
3. `N` means times that MoCo will execute mutate operation in one generation. Set it to an int.

4. `TSF` means "Train_Stop_Flag", set it to 0 or 1. If 1, MoCo will run without training. 

II. Boundary

Open your console under jittor_/algorithms and type this for MoCo BOUNDARY TEST:
```
python go_Boundary.py --MODEL 'lenet'
```
`MODEL` means the seed model used this time. Make sure that the model is in the list above,
and if you want to run with all seed models in a loop, set para MODEL to `LOOP`.

### OUTPUT
The result will be written in several directories
(if not exists, these directories will be created while running):

```jittor_/mutated_models``` -- models generated during running, but will be deleted every generation.

```jittor_/saved_mutated_models``` -- models that have some errors in it and have not passed the filter.

```jittor_/logs``` -- error logs will be recorded here.

```jittor_/boundary_logs``` -- boundary test result.

```jittor_/boundary_models``` -- models related to boundary test result.
