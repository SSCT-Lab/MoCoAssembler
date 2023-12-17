# MoCo for Jittor
### JITTOR VERSION
1.3.7.16

### REQUIREMENTS
Running MoCo_Jittor need some libs, just pip them:
```
pip install pyyaml
pip install alive_progress
pip install openpyxl
```

### DIRECTORY
All files about MoCo_Jittor are in this directory(moco_jt), but the datasets used for training are in the main directory.
There are several directories in jittor:

```moco_jt/algorithms``` -- algorithms.

```moco_jt/jittor_layer_info``` -- layers info which is written in yaml files.

```moco_jt/jittor_layer_similarity``` -- layers similarity info which is written in yaml files.

```moco_jt/seed_models``` -- seed model files.

And then, several directories will be created(auto) during running to contain running results.

### RUN
Due to the time cost of this tool, if you want to try this tool, here is a simple demo where less cases will be
generated:

Open your console under moco_jt/algorithms and type this for a demo:
```
python go_Fuzzing.py --MODEL 'LeNet' --N 3
```
I. Fuzzing

Open your console under moco_jt/algorithms and type this for MoCo FUZZING TEST:
```
python go_Fuzzing.py --MODEL 'LeNet' --N 5 --TSF 0
```
1. `MODEL` means the seed model used this time. Make sure that the model is in `
'alexnet'
'LeNet'
'resnet18'
'mobilenet'
'squeezenet'
'vgg19'
'lstm'
'googlenet'
'pointnet'`
. And if you want to run with all seed models in a loop, set para MODEL to `LOOP`.
2. `N` means times of mutation on every model in one generation. Set it to an int.

3. `TSF` means "Train_Stop_Flag", set it to 0 or 1. If 1, MoCo will run without training. 

II. Boundary

Open your console under moco_jt/algorithms and type this for MoCo BOUNDARY TEST:
```
python go_Boundary.py --MODEL 'LeNet'
```
`MODEL` means the seed model used this time. Make sure that the model is in the list above,
and if you want to run with all seed models in a loop, set para MODEL to `LOOP`.

### OUTPUT
The result will be written in several directories
(if not exists, these directories will be created while running):

```moco_jt/mutated_models``` -- models generated during running (will be deleted every generation).

```moco_jt/saved_mutated_models``` -- error models will be saved here, they may show some potential bugs of jittor.

```moco_jt/logs``` -- error logs will be recorded here.

```moco_jt/boundary_logs``` -- boundary test results.

```moco_jt/boundary_models``` -- models related to boundary test results.
