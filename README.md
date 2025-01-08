# MoCo：Fuzzing Deep Learning Libraries via Code Assembling


## Issue List
[Issue List](https://github.com/SSCT-Lab/MoCoAssembler/blob/MoCo_2.1/issue_list.md) stores the bug issues we submitted. 

### Description of README.md
> `DL Library`: Library name (`TensorFlow`, `PyTorch`).
> 
> `Issue`: Description of Bugs.
> 
> `URL`: Issue link of Bugs. 
> 
> `Status`: Issue Status of Bugs (`confirmed` or `fixed`). 
## Bug List
[Bug List](https://github.com/SSCT-Lab/MoCoAssembler/blob/MoCo_2.1/bugs) stores the bugs we can find. 
### Description of README.md
> `Bug API`: The API name that triggered the bug.
> 
> `Bug Type`: The type of bug that was triggered (`ICBug`, `BonBug`, `PerBug`, `ImpBug`).
> 
> `File Path`: Test case paths that can trigger bugs.
> 
> `Error Message`: Error description or error report.
## Directory structure

We provide specific `directory structures` according to different frameworks.

> `/MoCoAssembler/bugs` -- Bug list for MoCo. 
> 
> `/MoCoAssembler/moco_jt`  --  MoCo for jittor. 
>
> `/MoCoAssembler/moco_torch`  --  MoCo for pytorch. 
>
> `/MoCoAssembler/moco_tf`  --  MoCo for tensorflow. 
> 
> `/MoCoAssembler/datasets`  --  Datasets we used (Need to be added by the replicator). 
>
> `/MoCoAssembler/utils`  --  MoCo utility class. 
## Usage
In order to reproduce our tool, you first need to download and unzip `/MoCoAssembler`, and use MoCo according to the corresponding `README.md`.

| Library    | Link                                                                         |
| ---------- |------------------------------------------------------------------------------|
| Tensorflow | https://github.com/SATE-Lab/MoCoAssembler/blob/MoCo_2.1/moco_tf/README.md    |
| Pytorch    | https://github.com/SATE-Lab/MoCoAssembler/blob/MoCo_2.1/moco_torch/README.md |
| Jittor     | https://github.com/SSCT-Lab/MoCoAssembler/blob/MoCo_2.1/moco_jt/README.md    |
## Libraries
I. **Tensorflow**

[TensorFlow](https://github.com/tensorflow/tensorflow) is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.
TensorFlow was originally developed by researchers and engineers working within the Machine Intelligence team at Google Brain to conduct research in machine learning and neural networks. However, the framework is versatile enough to be used in other areas as well.

II. **Pytorch**

[PyTorch](https://github.com/pytorch/pytorch) is an open-source deep learning framework developed by Facebook's AI Research lab (FAIR). It is widely used in the research and development of artificial intelligence and machine learning applications. PyTorch provides a flexible and intuitive platform for building and training neural networks, making it a popular choice among researchers and developers.

III. **Jittor**

[Jittor](https://github.com/Jittor/jittor) is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. .

## Datasets

We used `9` deep learning models from `4` common datasets based on image and sequence data as the initial seed models for MoCo, and these models have been widely used in many existing studies.

| Model       | Dataset | Link                                                                                                                                |
| ----------- | ------- |-------------------------------------------------------------------------------------------------------------------------------------|
| AlexNet     | CIFAR-10 | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                                                                             |
| GoogLeNet   | ImageNet | [ImageNet](https://www.image-net.org/)                                                                                              |
| LeNet       | MNIST   | [MNIST](http://yann.lecun.com/exdb/mnist/)                                                                                          |
| MobileNet   | CIFAR-10 | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                                                                             |
| ResNet18    | ImageNet | [ImageNet](https://www.image-net.org/)                                                                                              |
| SqueezeNet  | ImageNet | [ImageNet](https://www.image-net.org/)                                                                                              |
| VGG19       | CIFAR-10 | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                                                                             |
| LSTM        | Stock-Price | [StockPricesPredictionProject](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject) |
| PointNet | Stock-Price| [StockPricesPredictionProject](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject)                                                |


I. We have made special treatment for different data sets, which are stored in the form of `.npz` for training and validation of the network model. 

II. We provide the processed dataset file [MoCo_Datasets](https://1drv.ms/f/s!Ao0nBM4MEX_uiU442vGWhqV05hwV?e=VONAIO).

**Note:** In datasets.zip, there are `4` files:

> cifar10.npz: dataset for cifar10;
>
> imagenet.npz: dataset for imagenet;
>
> mnist.npz: dataset for mnist;
>
> DIS.csv: dataset for DIS.
