# MOCO for Tensorflow

## Requirement

```
pyyaml
alive-progress
numpy
openpyxl
tensroflow==2.12.0
```

## Reproducibility

**STEP 0:** Download the environment requirements to run.

**Note: Please copy the installation command line by line to prevent some commands from being skipped.**

```
pip install tensroflow==2.12.0
pip install pyyaml
pip install alive-progress
pip install numpy
pip install openpyxl
```

**STEP 1:** Running MoCo_TF with fuzzing test.

Researchers can simply run MoCo_TF with the following command.

```
# Enter the tensorflow_/src environment
cd mnt/ModelAssembler/tensorflow_/src
```

I. Running a single model with fuzzing test. 

**Note: Researchers can query `tensorflow_/config/model.py` for the names of all the network models which can run.**

```
python mutate_tf.py --model_name "lenet" --mutate_times 3 
```

The initial value of `model_name` is `"lenet"`; The initial value of `mutate_time` is `3`, and the number of mutations we use in our formal experiments is `5`.  Also, It is possible to add the `--is_train` keyword to the command line to train the newly generated network model, The code is as follows:

```
python mutate_tf.py --model_name "lenet" --mutate_times 3 --is_train
```

II. Run all models with Fuzzing test. (Not recommended)

```
python mutate_tf.py --run_all True
```

It is also possible to add the `--is_train` keyword.

**STEP 2:** Running MoCo_TF with Boundary test.

**Note: Boundary testing does not support running all models.**

```
# Enter the tensorflow_/src environment
cd mnt/ModelAssembler/tensorflow_/src
python boundary_tf.py --model_name "lenet"
```

**Boundary tests can be run in isolation.** If the target model is not fuzzed, then when the boundary test is run, the fuzzing test is run with `mutate_time=3` and `is_train=False` before the boundary test

**STEP 3:** Obtain the output.

I. Obtain the output for fuzzing test.

`tensorflow_/result/mutate`: All models generated after fuzzing testing were performed. 

`tensorflow_/log`: After fuzzing runs, a log of those models that run the error.

`tensorflow_/log/model_name/log.csv`: All error logs. 

II. Obtain the output for boundary test.

`tensorflow_/result/boundary`: All models generated after boundary testing were performed. 

`tensorflow_/result/model_name_boundary_output.xlsx`: Results of boundary tests.

