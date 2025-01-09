# MoCo for Jittor
## JITTOR VERSION
1.3.7.16

## Requirements
```
pyyaml
alive-progress
numpy
openpyxl
jittor==1.3.7.16
```

## Datesets

All the datasets involved in jittor are stored in [MoCo_Datasets](https://1drv.ms/f/s!Ao0nBM4MEX_uiU442vGWhqV05hwV?e=VONAIO), unzip to the `/MoCoAssembler/moco_jt2/Data` directory, named `Datasets`.

You can check out the web datasets at the following link: [Cifar 10](https://www.cs.toronto.edu/~kriz/cifar.html), [Mnist](http://yann.lecun.com/exdb/mnist/), [Imagenet](https://www.image-net.org/), and [StockPricesPredictionProject](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject)

## Reproducibility

### **STEP 0:** Download the environment requirements to run.

**Note: Please copy the installation command line by line to prevent some commands from being skipped.**

```
pip install pyyaml
pip install alive-progress
pip install numpy
pip install openpyxl
pip install jittor==1.3.7.16
```

### **STEP 1:** Running MoCo_JT with Fuzzing test.

Researchers can simply run MoCo_JT with the following command.

```
# Enter the moco_jt2 environment
cd /MoCoAssembler/moco_jt2
```

I. Running a demo test with Fuzzing.

```
python MoCoFuzzing.py --model "lenet" --n 3 --max 500 --output "lenet"
```

The initial value of `model` is `"lenet"`; The initial value of `n` is `3`, and the number of mutations we use in our formal experiments is `5`; The max layer nodes value of `max` is `500`.

II Run a single model with Fuzzing test.
```
python MoCoFuzzing.py --model "lenet" --n 4 --max 500 --output "lenet"
```

### **STEP 2:** Running MoCo_JT with Boundary test.

**Note: Boundary testing does not support running all models.**

```
# Enter the moco_jt2 environment
cd /MoCoAssembler/moco_jt2
python MoCoBoundary.py --model "lenet" --output "lenet_b"
```

### **STEP 3:** Obtain the output.

I. Obtain the output for Fuzzing test.

`moco_jt2/output/*/tree`: All models generated after Fuzzing testing were performed. 

`moco_jt2/output/*/report`: After Fuzzing runs, a log of those models that run the error.

II. Obtain the output for boundary test.

`moco_jt2/output/*/tree`: All models generated after boundary testing were performed.



