# MoCoAssembler

## Introduction
The rapidly developing Deep Learning (DL) techniques have been applied in software systems belonging to different application scenarios. While these DL systems bring convenience to people’s lives, they also bring new safety threats with potentially serious consequences, especially in safety-critical domains. Currently, a lot of research has focused on how to test DL models or domain-specific DL applications, and less research has focused on how to test DL libraries. DL libraries serve as the underlying foundation for DL systems, and bugs in them can have unpredictable impacts that directly affect the behaviors of DL systems. In this paper, we propose a novel fuzzing testing method for DL libraries and implement it in the testing tool MoCoAssembler. To solve the lack of high-quality tests, we design a code assembling method to simulate the process of constructing, training, and evaluating DL models in real-world user scenarios. The method first disassembles the seed code file for building DL models to obtain the template and code blocks and then employs the code block mutation strategies to generate more new code blocks adapted to the template. By inserting context-appropriate code blocks into the template in step, we can obtain a tree of code files with intergenerational relations, where each node represents a code file that should work properly for building a DL model. Therefore, if a node is not working successfully, it is easy to locate which code block the bug may be in. We conducted a comprehensive experiment to evaluate the efficiency and effectiveness of MoCoAssemblerwith three widely-used DL libraries, i.e., TensorFlow, PyTorch and Jittor. During the experiments, MoCoAssembler detects 57 new bugs of four types in the latest release versions of three DL libraries, where 49 bugs have been confirmed and 8 bugs have been fixed by developers. The experimental results demonstrate that MoCoAssembler is capable of generating high-quality tests and detecting different types of bugs to help developers improve the reliability of DL libraries.

## Bug List

Check out [BUG_LIST](https://github.com/SATE-Lab/MoCoAssembler/blob/main/bug_list.md) for a summary of the bugs our tool has detected.

## Libraries
I. **Tensorflow**

[TensorFlow](https://github.com/tensorflow/tensorflow) is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.
TensorFlow was originally developed by researchers and engineers working within the Machine Intelligence team at Google Brain to conduct research in machine learning and neural networks. However, the framework is versatile enough to be used in other areas as well.

II. **Pytorch**

[PyTorch](https://github.com/pytorch/pytorch) is an open-source deep learning framework developed by Facebook's AI Research lab (FAIR). It is widely used in the research and development of artificial intelligence and machine learning applications. PyTorch provides a flexible and intuitive platform for building and training neural networks, making it a popular choice among researchers and developers.

III. **Jittor**

[Jittor](https://github.com/Jittor/jittor) is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. .

## Dataset/directories

### Datasets

We used `9` deep learning models from `5` common datasets based on image and sequence data as the initial seed models for MoCo, and these models have been widely used in many existing studies.

| Model       | Dataset     | Link                                                         |
| ----------- | ----------- | ------------------------------------------------------------ |
| AlexNet     | CIFAR-10    | [cifar 10](https://www.cs.toronto.edu/~kriz/cifar.html)      |
| GoogLeNet   | ImageNet    | [Imagenet](https://www.image-net.org/)                       |
| LeNet       | MNIST       | [mnist](http://yann.lecun.com/exdb/mnist/)                   |
| MobileNet   | CIFAR-10    | [cifar 10](https://www.cs.toronto.edu/~kriz/cifar.html)      |
| ResNet18    | ImageNet    | [Imagenet](https://www.image-net.org/)                       |
| SqueezeNet  | ImageNet    | [Imagenet](https://www.image-net.org/)                       |
| VGG19       | CIFAR-10    | [cifar 10](https://www.cs.toronto.edu/~kriz/cifar.html)      |
| LSTM        | Stock-Price | [StockPricesPredictionProject](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject) |
| PointNet | | |


I. We have made special treatment for different data sets, which are stored in the form of `.npz` for training and validation of the network model. 

II. We provide the processed dataset file [MoCo_Datasets](https://1drv.ms/f/s!Ao0nBM4MEX_uiU442vGWhqV05hwV?e=VONAIO).

**Note:** In datasets.zip, there are 4 files:

> cifar10.npz: dataset for cifar10;
>
> imagenet.npz: dataset for imagenet;
>
> mnist.npz: dataset for mnist;
>
> DIS.csv: dataset for DIS.

### Directory structure

We provide specific `directory structures` according to different frameworks.

> ```/MoCoAssembler/jittor```  --  MoCo for jittor.
>
> ```/MoCoAssembler/pytorch```  --  MoCo for pytorch.
>
> ```/MoCoAssembler/tf```  --  MoCo for tensorflow.
>
> ```/MoCoAssembler/examples```  --  Some models generated by MoCo as examples.
>
> ```/MoCoAssembler/datasets```  --  Datasets we used.
>
> ```/MoCoAssembler/utils```  --  MoCo utility class.

## Usage

To use our tool, please download and unzip it first. We provide different ways to use different libraries. In addition, we have provided three detailed usage documents for your reference.

| Library    | Link                                                         |
| ---------- | ------------------------------------------------------------ |
| Tensorflow | https://github.com/SATE-Lab/MoCoAssembler/blob/main/tf/README.md |
| Pytorch    | https://github.com/SATE-Lab/MoCoAssembler/blob/main/pytorch/README.md |
| Jittor     | https://github.com/SATE-Lab/MoCoAssembler/blob/main/jittor/README.md |
