# Author: .
#
# Modified from J.-H. Jacobsen code.
# Original license shown below.
# =============================================================================
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import paddle
import paddle.fluid as fluid
from paddle.fluid import layers, dygraph as dg
from models.model_utils import get_all_params, CrossEntropyLoss

import os
import sys
import math
import numpy as np

fluid.enable_dygraph()

criterion = CrossEntropyLoss()

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def train(model, trainloader, trainset, epoch, num_epochs, batch_size, lr, use_cuda, in_shape):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = fluid.optimizer.SGD(learning_rate(lr, epoch), model.parameters())

    model_parameters = filter(lambda p: p.stop_gradient, model.parameters())
    params = sum([np.prod(p.shape) for p in model_parameters])

    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.clear_gradients()
        out, out_bij = model(inputs)               # Forward Propagation
        loss = criterion(out, targets)  # Loss
        if use_cuda and model._strategy and model._is_data_parallel_mode():
            loss = model.scale_loss(loss) # scale the loss for data parallel
        loss.backward()  # Backward Propagation
        if use_cuda and model._strategy and model._is_data_parallel_mode():
            model.apply_collective_grads() # collect the gradients for data parallel
        optimizer.minimize(loss)  # Optimizer update

        train_loss += loss.numpy()[0]
        predicted = layers.argmax(out, axis=1)
        total += targets.shape[0]
        correct += layers.equal(predicted, targets).numpy().astype('float32').sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.numpy()[0], 100.*correct/total))
        sys.stdout.flush()


def test(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with dg.no_grad():
            out, out_bij = model(inputs)
            loss = criterion(out, targets)
            if use_cuda and model._strategy and model._is_data_parallel_mode():
                loss = model.scale_loss(loss) # scale the loss for data parallel

        test_loss += loss.numpy()[0]
        predicted = layers.argmax(out, axis=1)
        total += targets.shape[0]
        correct += layers.equal(predicted, targets).numpy().astype('float32').sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.numpy()[0], acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        dg.save_dygraph(model.state_dict(), save_point+fname+f'.epoch_{str(epoch).zfill(3)}.acc_{acc:.3f}')
        best_acc = acc
    return best_acc
    
    
def invert(model, test_loader, mean, std):
    model.eval()
    for i, (inputs, targets) in enumerate(test_loader):

        # compute output
        output, out_bij = model(inputs)

        # invert bijective output
        x_inv = model._layers.inverse(out_bij)

        inp = inputs[:8,:,:,:]
        x_inv = x_inv[:8,:,:,:]

        grid = layers.concat((inp, x_inv),2).numpy()
        mean= np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        grid = grid[:,:,:,:] * std[None, :, None, None]
        grid = grid[:,:,:,:] + mean[None, :, None, None]
        grid *= 255.
        grid = np.uint8(grid)
        grid = grid.transpose((0, 3, 2, 1)).reshape((-1, 32*2, 3)).transpose((1, 0, 2))
        import matplotlib.pyplot as plt
        plt.imsave('invert_eval_samples.jpg', grid)
        return
    