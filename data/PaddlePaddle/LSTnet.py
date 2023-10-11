import paddle


class LstNet(paddle.nn.Layer):
    # 接收参数构建网络
    # P是window, m是输入时间序列变量的维度数, hidR是RNN的隐藏层维度，hidC是CNN的隐藏层维度,hidS是hidSkip,Ck是cnn的卷积核大小，skip是跳跃的步数，pt = (window - 卷积核size) // 跳跃的步数
    # hw 是 The window size of the highway component
    def __init__(self, P, m, hidR, hidC, hidS, Ck, skip, hw, output_fun):
        super(LstNet, self).__init__()
        self.pt = int((P - Ck) // skip)
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=hidC, kernel_size=(Ck, m))
        self.GRU1 = paddle.nn.GRU(input_size=hidC, hidden_size=hidR)
        self.dropout = paddle.nn.Dropout()
        self.relu = paddle.nn.ReLU()
        self.skip = skip
        self.hidC = hidC
        self.hidS = hidS
        self.hw = hw
        self.m = m

        if skip > 0:
            self.GRUskip = paddle.nn.GRU(input_size=hidC, hidden_size=hidS)
            self.linear1 = paddle.nn.Linear(in_features=hidR + skip * hidS, out_features=m)
        else:
            self.linear1 = paddle.nn.Linear(in_features=hidR, out_features=m)
        if hw > 0:
            self.highway = paddle.nn.Linear(in_features=hw, out_features=1)
        self.output = None
        if output_fun == 'sigmod':
            self.output = paddle.nn.Sigmoid()
        if output_fun == 'tanh':
            self.output = paddle.nn.Tanh()

    def forward(self, x):
        # x的输入应该是（N,1，P，m）,1是Chanel，P是window,m是时间序列的变量数
        # CNN部分

        batch_size = x.shape[0]
        c = self.conv1(x)
        print("c.shape=", c.shape)
        c = self.relu(c)
        c = self.dropout(c)  # c.shape= [4, 50, 163, 1]

        c = paddle.squeeze(c, axis=3)  # 把最后一维1删掉   [4, 50, 55]
        # RNN部分
        r = paddle.transpose(c, (2, 0, 1))
        print("r.shape1=", r.shape)  # r.shape1= [163, 4, 50]
        # gru输入：则Tensor的形状为[batch_size,time_steps,input_size]。
        # 这里需要把数据的维度做一下转换，因为pytorch默认batch_first = false,与paddle不同
        r = paddle.transpose(r, (1, 0, 2))
        _, r = self.GRU1(r)  # [num_layers * num_directions, batch_size, hidden_size]
        print("r.shape2=", r.shape)  # r.shape2= [1, 4, 50]
        r = self.dropout(paddle.squeeze(r, 0))
        print("r.shape3=", r.shape)  # r.shape3= [4, 50] 把1的维度删掉了，因为num_layers * num_directions=1

        # skip-rnn
        # 将进行一次卷积的数据输入到skip-rnn里面去
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):]
            print("s.shape1=", s.shape)  # s.shape1= [4, 50, 144]   # 把第一步从卷积网络中截掉了一小段

            s = s.reshape((batch_size, self.hidC, self.pt, self.skip))
            print("s.shape2=", s.shape)  # s.shape2= [4, 50, 6, 24]   2 * 24 = 48 正好可以将最后一维48分离成 2 * 24， 将

            s = paddle.transpose(s, (2, 0, 3, 1))
            print("s.shape3=", s.shape)  # s.shape3= [6, 4, 24, 50]

            s = s.reshape((self.pt, batch_size * self.skip, self.hidC))
            print("s.shape4=", s.shape)  # s.shape4= [6, 96, 50]  (seq, batch, feature)
            # 这里需要把数据的维度做一下转换，因为pytorch默认batch_first = false,与paddle不同
            s = paddle.transpose(s, (1, 0, 2))
            _, s = self.GRUskip(s)  # 输入 [batch_size,time_steps,input_size]。
            print("s.shape5=",
                  s.shape)  # s.shape5= [1, 96, 5]  # [num_layers * num_directions, batch_size, hidden_size]

            s = s.reshape((batch_size, self.skip * self.hidS))
            print("s.shape6=", s.shape)  # s.shape6= [4, 120]

            s = self.dropout(s)
            print("s.shape7=", s.shape)  # s.shape7= [4, 120]

            r = paddle.concat((r, s), 1)
            print("r.shape4=", r.shape)

        #
        res = self.linear1(r)
        print("res.shape=", res.shape)

        if self.hw > 0:
            print("x.shape=", x.shape)
            x = paddle.squeeze(x, axis=1)
            z = x[:, -self.hw:, :]
            print("z.shape=", z.shape)  # z.shape= [4, 24, 8]
            z = paddle.transpose(z, (0, 2, 1)).reshape((-1, self.hw))
            print("z.shape1=", z.shape)  # z.shape1= [32, 24]
            z = self.highway(z)
            print("z.shape2=", z.shape)  # z.shape2= [32, 1]
            z = z.reshape((-1, self.m))
            print("z.shape3=", z.shape)  # z.shape3= [4, 8]
            res = res + z
            print("res1.shape=", res.shape)  # res1.shape= [4, 8]

        if self.output:
            res = self.output(res)

        return res

