# 修改说明（代码处理需注意的部分）
1. SynBatchNorm 中 process_group 的类型是 Optional[Any]，建议直接删除这个可选参数
2. LayerNorm 中 normalized_shape 的 shape 是 ANY，意思是list长度可以为任何正整数（原理是最后的数字为lst每个元素的乘积），建议限制类型为int，不需要保留list类型
3. Threshold 中 两个参数本身应该不存在range（可以取任意float值，具体原因见函数定义）
4. MaxPool1d/2d 中 padding 的 range 为 [0, ks/2]，指最大值为 kernel_size 取值的一半
5. TransformerDecoder/TransformerEncoder 的decoder_layer/encoder_layer 的类型为其对应的 xxxLayer, 具体代码处理时需注意
6. Upsample 的 shape 为 {1,2,3} 指可以为1或2或3
7. Softmax/Softmin/LongSoftmax 涉及到dim参数，参数应该有限制 LIMITED，这里删除了

# 需要注意的地方
1. 若出现找不到yaml文件中某属性，有可能是单词拼错了等低级错误，我筛查过一遍应该问题不大
2. Upsamplingxxx2d，ZeroPad2d没有对应的1d/3d，其对应的应该就是Upsample
3. 在官网上没有找到padding可以为enum的情况...