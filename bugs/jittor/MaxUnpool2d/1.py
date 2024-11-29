import jittor as jt
import jittor.nn as nn

# 创建输入数据
input_data = jt.array([[[[1, 2, 3, 4, 5, 6]]]])

# 最大池化操作，获取输出和索引
max_pool = nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 1), return_indices=True)
output, indices = max_pool(input_data)

# 最大反池化操作
unpool = nn.MaxUnpool2d(kernel_size=(1, 6), stride=(8, 3))
unpooled_output = unpool(output)