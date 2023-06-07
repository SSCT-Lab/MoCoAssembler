# PyTorch常见API解释



## init 部分

### Tensor的四维

* $(N,C,W,H)$，分别为 **batch_size** (N), **channel** (C), **width** (W), **height** (H)

### activation
```python
self.relu = nn.ReLU(inplace=True)
self.relu = nn.ReLU()
self.softmax = nn.Softmax(dim=1)
```

* nn.ReLU()函数默认inplace**默认是False**，False会新创建一个对象，True则在输入的对象上修改并返回
* nn.Softmax(dim=k)表示从外到内的第k层进行softmax

### pool

```python
self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
self.avgpool = nn.AdaptiveAvgPool2d(1)
```

* MaxPool2d参数解释：

  * `MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`

  * 常用参数

    > kernel_size(int or tuple) - max pooling的**窗口大小**
    >
    > stride(int or tuple, optional) - max pooling的窗口移动的**步长**。**默认值是kernel_size**
    >
    > padding(int or tuple, optional) - 输入的每一条边**补充0的层数**

  * 不常用参数

    > dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
    >
    > return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
    >
    > ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

  * 对输出维度的影响
    $$
    H_{out} = \frac{H_{in}-K+2P}{S}+1
    $$

* AdaptiveAvgPool2d

  * `nn.AdaptiveAvgPool2d(output_size)`

  * 参数即输出维度，可以为一个数或二元元组，$L_{out} = output\_size[0], \ H_{out} = output\_size[1] $

### batchnorm

```python
self.bn32 = nn.BatchNorm2d(32)
```

* 卷积层之后总会添加BatchNorm2d进行**数据的归一化处理**，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
* 参数解释：`nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)`
  * 通常BatchNorm2d的参数`num_features`取`channel`值，即tensor第二维

### conv

```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
```

* 参数解释：`nn.Conv2d(in_channels，out_channels，kernel_size，stride=1，padding=0，dilation=1，groups=1，bias=True)`

  * 常用参数

    > in_channels：输入维度
    >
    > out_channels：输出维度
    >
    > kernel_size：卷积核大小（两种表示方式，一个数则为正方形kernel，二位元组则为长方形kernel）
    >
    > stride：步长大小
    >
    > padding：填充的列/行数，注意实际值需要乘2

  * 不常用的几个参数

    > dilation = 1，决定是否采用空洞卷积**，**默认为1（不采用）
    >
    > groups = 1，决定了是否采用分组卷积
    >
    > bias = True，是否要添加偏置参数作为可学习参数的一个，默认为True

* 假设输入的张量为`(2,C,L,W)`，调用语句为`nn.Conv2d(I, O, kernel_size=(K1,K2), stride=(S1,S2), padding=(P1,P2))`

  * 需要满足：$C = I$，即输入维度对齐

  * 输出张量为
    $$
    (2, O,\frac{L-K_1+2P_1}{S_1}+1,\frac{H-K_2+2P_2}{S_2}+1)
    $$
    

### dropout

```python
self.dropout = nn.Dropout(p=dropout)
```

* **以概率P随机的将参数置0**，其中P为置0的概率，例如P=1表示将网络参数全部置0
* 输出的参数会以$\frac{1}{1-p}$的比例扩大

### linear

```python
self.linear1 = nn.Linear(256 * 6 * 6, 4096)
self.rx_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
```

* 参数解释 `nn.Linear(in_features, out_features, bias=True)`
  * in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size
  * out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数
* 通常需要**先转换为二维向量**然后使用该函数，需要保证最后一位与`in_features`相等

### rnn

```python
self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
```

### sparse
```python
self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
```

* input_size个词，每个词用hidden_dim维词向量表示



## forward 部分

```python
out = out.view(self.sequence_len, x.size(0), -1)
x = x.view(-1, 512 * 7 * 7)
x = torch.flatten(x, 1)
```

* view的作用相当于numpy中的reshape，**重新定义矩阵的形状**。
  * 需要保证前后两个矩阵中元素的总数相等
  * 不确定的位置可以用**-1**填充（只能推断1位），系统会自动推断
    * 若将推断出的为小数，则报错
    * 若不变，则可以填写`x.size(k)`
* x = torch.flatten(x, k)，从第k位（k从0开始）往后塌缩为1位（保留k+1位）

```python
x = torch.cat(outputs)
input_tensor = torch.cat((fwd, bwd), 1)
x = torch.concat(features_list, dim=-1)
torch.unsqueeze(layer, 0)
```
* cat函数即连接列表/元组中的元素，两个参数：list和dim
  * 连接后$x.size(dim) = \sum_{p} x_p.size(dim)$
    * dim默认为0，可以看作是去掉最外围的括号，合并到一起（从shape变化角度理解）
    * dim为-1则为最后一位
  * concat函数即为cat函数（别名）
  
* torch.unsqueeze()函数起到**升维的作用**，dim等于几表示在第几维度加一

```python
hs_forward = torch.zeros(x.size(0), self.hidden_dim)
torch.randn(x.shape[1], self.hidden_dim)
```
* torch.zeros按照后面的size生成每一位都为0的tensor
  * torch.ones意义类似，生成的每一位都为1

* torch.randn生成size指明的tensor，每一位的数字符合正态分布

```python
r = torch.sigmoid(r)
h_ = torch.tanh(h_)
```

* 激活函数
  $$
  sigmoid \Rightarrow \ out_i = \frac{1}{1+e^{-in_i}}
  $$

  $$
  tanh \Rightarrow \ out_i = \frac{e^{in_i}-e^{-in_i}}{e^{in_i}+e^{-in_i}}
  $$

  

```python
torch.nn.init.kaiming_normal_(hs_forward)
```

* torch.nn.init 初始化函数系列

  > 1. 均匀分布  torch.nn.init.**uniform_**(tensor, a=0, b=1)，例如 nn.init.**uniform_**(w)
  > 2. 正态分布  torch.nn.init.**normal_**(tensor, mean=0, std=1)，例如 nn.init.**normal_**(w)
  > 3. 常数固定值  torch.nn.init.**constant_**(tensor, val)，例如 nn.init.**constant_**(w, 0.3)
  > 4. 主对角线为1（长方形也可，从左上角开始）torch.nn.**init.eye_**(tensor)，例如 nn.init.**eye_**(w)
  > 5. delta函数初始化，仅适用于3, 4, 5维  torch.nn.init.**dirac_**(tensor)，例如 nn.init.**dirac_**(w)
  > 6. xavier_uniform初始化  torch.nn.init.**xavier_uniform_**(tensor, gain=1)，例如 nn.init.**xavier_uniform_**(w, gain=nn.init.calculate_gain('relu'))
  > 7. xavier_normal初始化 torch.nn.init.**xavier_normal_**(tensor, gain=1)，例如 nn.init.**xavier_normal_**(w)
  > 8. kaiming_uniform初始化  torch.nn.init.**kaiming_uniform_**(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')，例如 nn.init.**kaiming_uniform_**(w, mode='fan_in', nonlinearity='relu')
  > 9. kaiming_normal初始化 torch.nn.init.**kaiming_normal_**(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')，例如 nn.init.**kaiming_normal_**(w, mode='fan_out', nonlinearity='relu')
  > 10. 正交矩阵 torch.nn.init.**orthogonal_**(tensor, gain=1)，例如 nn.init.**orthogonal_**(w)
  > 11. 稀疏矩阵 非零元素采用正态分布 N(0, 0.01) 初始化  torch.nn.init.**sparse_**(tensor, sparsity, std=0.01)，例如nn.init.**sparse_**(w, sparsity=0.1)
