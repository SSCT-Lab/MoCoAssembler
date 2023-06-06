# ModelAssembler for Jittor
##### 需要下载的工具
```
pip install yaml
pip install jittor
pip install spacy
pip install sentence_transformers
```

##### 启动
在jittor_/algorithms文件夹下开启终端然后输入以下命令
```
python demo.py --MODEL 'testnet' --N 2
```

##### 注意事项
demo是一个单个模型变异的文件，MODEL表示选择的模型，N表示每层变异多少次
使用testnet可以很快输出一个具有代表性的结果，N最好设置为2
N设置为3或更多会使模型数量指数级增加，演示时建议N=2

如果想使用其他模型演示，请在以下MODEL中选择
'testnet'
'alexnet'
'lenet'
'ResNet18'
'ResNet50'
'nasnet'
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

第一次变异时会加载算子，后面就快了

报错信息会保存在log.txt，变异后的模型保存在mutated_model下，出错模型保存在saved_mutated_models下并附上时间