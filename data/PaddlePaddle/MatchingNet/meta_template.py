import paddle
import paddle.nn as nn
import numpy as np
from abc import abstractmethod
import paddle.optimizer as optim


class MetaTemplate(nn.Layer):
    def __init__(self, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self, x):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        if is_feature:
            z_all = x
        else:
            x = x.reshape([self.n_way * (self.n_support + self.n_query), *x.shape[2:]])
            z_all = self.feature.forward(x)
            z_all = z_all.reshape([self.n_way, self.n_support + self.n_query, -1])
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x).detach()
        y_query = np.repeat(range(self.n_way), self.n_query)
        topk_scores, topk_labels = scores.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, logger, optimizer):
        print_freq = 10

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.shape[1] - self.n_support
            if self.change_way:
                self.n_way = x.shape[0]
            optimizer.clear_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                logger.info('Epoch {:d} | Batch {:d}/{:d} | loss {:f} | avg_loss {:f}'.format(epoch, i, len(train_loader),
                                                                                              loss.item(), avg_loss / float(i + 1)))

    def test_loop(self, test_loader, logger, record=None):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.shape[1] - self.n_support
            if self.change_way:
                self.n_way = x.shape[0]
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        logger.info('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean