import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
	# B,G,R 三个通道归一化 标准差为 0.5， 方差为0.5
    [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car','bird','cat','deer','dog','frog','horse','ship','truck')



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

from new_models.new_20  import KitModel
net =KitModel()
#
# net = net.to(device)
# print(net)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #3*32*32 ==》6*28*28(32 + 2P - kernel + 1)
        self.pool = nn.MaxPool2d(2, 2) #6*28*28 ==> 6*14*14
        self.conv2 = nn.Conv2d(6, 16, 5) # 6*14*14 ==> 16 * 10 * 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 还要一个pooling 所以输入是16 * 5 * 5 ==> 120
        self.fc2 = nn.Linear(120, 84) # 120 ==> 84
        self.fc3 = nn.Linear(84, 10) # 84 ==> 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from new_models.new_2 import KitModel
net =KitModel()

# 优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

print('---------- Train Start ----------')
epochs =1
test_iter = 0
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0),total =len(trainloader),leave = True):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    #没2000个小批次打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('----------Finished Training----------')
