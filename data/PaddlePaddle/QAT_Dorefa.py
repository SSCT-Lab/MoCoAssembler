import paddle
import paddleslim as slim
import numpy as np
from paddle.optimizer import Adam
from paddle.fluid import core
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import ConvertToInt8Pass
paddle.enable_static()


# 采用cpu训练则取False
USE_GPU = True
model = slim.models.MobileNet()
train_program = paddle.static.Program()
startup = paddle.static.Program()
with paddle.static.program_guard(train_program, startup):
    image = paddle.static.data(
        name='image', shape=[None, 1, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    gt = paddle.reshape(label, [-1, 1])
    out = model.net(input=image, class_dim=10)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=gt)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=gt, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=gt, k=5)
    opt = paddle.optimizer.Momentum(0.01, 0.9)
    opt.minimize(avg_cost)

place = paddle.CUDAPlace(0) if USE_GPU else paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(startup)
scope = paddle.static.global_scope()
val_program = train_program.clone(for_test=True)

import paddle.vision.transforms as T
transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
# Paddle框架的paddle.vision.dataset包定义了MNIST数据的下载和读取
train_dataset = paddle.vision.datasets.MNIST(
    mode="train", backend="cv2", transform=transform)
test_dataset = paddle.vision.datasets.MNIST(
    mode="test", backend="cv2", transform=transform)
train_loader = paddle.io.DataLoader(
    train_dataset,
    places=place,
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=True)
test_loader = paddle.io.DataLoader(
    test_dataset,
    places=place,
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=False)

# 训练和测试
outputs = [acc_top1.name, acc_top5.name, avg_cost.name]
def train(prog):
    iter = 0
    for data in train_loader():
        acc1, acc5, loss = exe.run(prog, feed=data, fetch_list=outputs)
        if iter % 100 == 0:
            print('train iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        iter += 1

def test(prog):
    iter = 0
    res = [[], []]
    for data in test_loader():
        acc1, acc5, loss = exe.run(prog, feed=data, fetch_list=outputs)
        if iter % 100 == 0:
            print('test iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        res[0].append(acc1.mean())
        res[1].append(acc5.mean())
        iter += 1
    print('final test result top1={}, top5={}'.format(np.array(res[0]).mean(), np.array(res[1]).mean()))

# train(train_program)
# test(val_program)


quant_config = {
    'weight_quantize_type': 'dorefa',
    #'activation_quantize_type': 'dorefa',
    'weight_bits': 8,
    'activation_bits': 8,
    'not_quant_pattern': ['skip_quant'],
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9
}
#quant_exe = paddle.static.Executor(place)

def create_tmp_var(program, name, dtype, shape):
    return program.current_block().create_var(name=name, dtype=dtype, shape=shape)

# 权重部分量化函数
def _weight_dorefa_quantize_func_forward(in_node):
    '''
    Weight forward function of derefa method.
    '''
    weight_bits = quant_config['weight_bits']
    input = np.array(in_node)
    output_mid = np.tanh(input)
    output_mid = output_mid / 2 / np.max(np.abs(output_mid)) + 0.5
    scale = 1 / float((1 << (weight_bits - 1)) - 1)
    output_mid = np.round(output_mid / scale) * scale   # STE
    output = 2 * output_mid - 1
    return output

def _weight_dorefa_quantize_func_backward(input, output, output_grad):
    '''
    Weight backward function of derefa method.
    '''
    input = np.array(input)
    output = np.array(output)
    output_grad = np.array(output_grad)
    return output / input * output_grad

def _weight_dorefa_quantize_func(in_node):
    '''
    Use Dorefa method to quantize weight.
    '''
    var_name = in_node.name[0: len(in_node.name) - 10]
    out_node_name = var_name + '_tmp_output'

    out_node = create_tmp_var(paddle.static.default_main_program(),
                              name=out_node_name,
                              dtype='float32',
                              shape=in_node.shape)
    
    paddle.static.nn.py_func(func=_weight_dorefa_quantize_func_forward,
                             x=in_node,
                             out=out_node,
                             backward_func=_weight_dorefa_quantize_func_backward
                             )
    return out_node

# 激活部分量化函数
def _act_dorefa_quantize_func_forward(in_node):
    '''
    Activation forward function of derefa method.
    '''
    activation_bits = quant_config['activation_bits']
    input = paddle.clip(in_node * 0.1, 0, 1)
    input = np.array(input)
    scale = 1 / float((1 << (activation_bits - 1)) - 1)
    output = np.round(input / scale) * scale   # STE
    return output

def _act_dorefa_quantize_func_backward(output_grad):
    '''
    Activation backward function of derefa method.
    '''
    return np.array(output_grad)

def _act_dorefa_quantize_func(in_node):
    '''
    Use Dorefa method to quantize activation.
    '''
    var_name = in_node.name[0: len(in_node.name) - 10]
    out_node_name = var_name + '_tmp_output'

    out_node = create_tmp_var(paddle.static.default_main_program(),
                              name=out_node_name,
                              dtype='float32',
                              shape=in_node.shape)
    
    paddle.static.nn.py_func(func=_act_dorefa_quantize_func_forward,
                             x=in_node,
                             out=out_node,
                             #backward_func=None
                             backward_func=_act_dorefa_quantize_func_backward,
                             skip_vars_in_backward_input=[in_node, out_node]
                             )
    return out_node

quant_program = slim.quant.quant_aware(train_program,
                                       exe.place,
                                       quant_config,
                                       scope,
                                       for_test=False,
                                       weight_quantize_func=_weight_dorefa_quantize_func,
                                       #act_quantize_func=_act_dorefa_quantize_func,
                                       optimizer_func=Adam,
                                       executor=exe)
val_quant_program = slim.quant.quant_aware(val_program,
                                           exe.place,
                                           quant_config,
                                           scope,
                                           for_test=True,
                                           weight_quantize_func=_weight_dorefa_quantize_func,
                                           #act_quantize_func=_act_dorefa_quantize_func,
                                           optimizer_func=Adam,
                                           executor=exe)
# 量化后测试，并与前测试比较精度
train(quant_program)
test(val_quant_program)

# 转换函数
def _dorefa_convert(program, place):

    test_graph = IrGraph(core.Graph(program.desc), for_test=True)
    convert_int8_pass = ConvertToInt8Pass(scope=scope, place=place)
    convert_int8_pass.apply(test_graph)
    program_int8 = test_graph.to_program()
    return program_int8

# 产出量化模型
quant_infer_program = _dorefa_convert(val_quant_program, exe.place)
target_vars = [quant_infer_program.global_block().var(outputs[-1])]
paddle.static.save_inference_model(
        path_prefix='./quant_dorefa_infer_model',
        feed_vars=[image],
        fetch_vars=target_vars,
        executor=exe,
        program=quant_infer_program)