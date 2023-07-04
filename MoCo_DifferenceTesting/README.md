# MoCo_DifferenceTesting

### 需要下载的工具
首先确保环境中已经安装好、配置好计图、pytorch框架，使其可以正常运行，然后安装以下工具：
```
pip install alive_progress
```

### 启动
在MoCo_DifferenceTesting文件夹下开启终端然后输入以下命令
```
python DifferenceTesting.py --MODEL 'TestNet'
```
即可以一个简单的TestNet为模板进行差异测试，并输出一批具有代表性的测试结果

### 注意事项
MODEL表示本次差异测试选择的模板。
使用TestNet可以输出一个简单的、具有代表性的结果，以方便演示

如果想使用其他模型演示，请在以下MODEL中选择
'LeNet'
'AlexNet'
'ResNet18'
'ResNet50'
'InceptionV3'
'VGG16'
'VGG19'
'SqueezeNet'
'MobileNet'
'GoogleNet'

第一次变异时会加载算子，后面就快了

该系统通过代码组装、测试输入、对比，来对不同的框架进行差异测试。
测试过程中，对两个框架并行组装，并不断使用若干输入进行测试，然后对比不同框架下，同样输入、同样功能代码产生的结果

### 结果
系统会将异常结果保存在MoCo_DifferenceTesting/results文件夹中，其中：

1.对于在相同输入、相同功能代码下，输出产生差异的情况，判定为严重差异，将组装的源代码保存在MoCo_DifferenceTesting/results/saved_models中，
同时将log写入MoCo_DifferenceTesting/results/os_log.txt

2.对于在相同输入、相同功能代码下，错误检测产生差异的情况，判定为严重差异，将组装的源代码保存在MoCo_DifferenceTesting/results/saved_models中，
同时将log写入MoCo_DifferenceTesting/results/ed_log.txt

3.对于在相同输入、相同功能代码下，错误报告产生差异的情况，判定为待定差异，仅将log写入MoCo_DifferenceTesting/results/es_log.txt