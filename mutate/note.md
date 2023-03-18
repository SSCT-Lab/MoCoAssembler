# 常见api

## init 部分
```python
self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
self.avgpool = nn.AdaptiveAvgPool2d(1)

self.relu = nn.ReLU(inplace=True)
self.relu = nn.ReLU()
self.softmax = nn.Softmax(dim=1)
self.dropout = nn.Dropout(p=dropout)

self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)

self.linear1 = nn.Linear(256 * 6 * 6, 4096)
self.rx_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)

self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
self.bn32 = nn.BatchNorm2d(32)
```

## forward 部分
```python
x = torch.cat(outputs)
input_tensor = torch.cat((fwd, bwd), 1)
x = torch.concat(features_list, dim=-1)

torch.randn(x.shape[1], self.hidden_dim)
r = torch.sigmoid(r)
h_ = torch.tanh(h_)

torch.unsqueeze(layer, 0)
x = torch.flatten(x, 1)
hs_forward = torch.zeros(x.size(0), self.hidden_dim)

nn.init.kaiming_normal_(hs_forward)
out = out.view(self.sequence_len, x.size(0), -1)
x = x.view(-1, 512 * 7 * 7)
```