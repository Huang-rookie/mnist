import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy
input_size = 28*28 # MNIST上的图像尺寸是 28x28
output_size = 10  # 类别为 0 到 9 的数字，因此为10类

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=1000, shuffle=True)
class FC2Layer(nn.Module):
  def __init__(self, input_size, n_hidden, output_size):
    # nn.Module子类的函数必须在构造函数中执行父类的构造函数
    # 下式等价于nn.Module.__init__(self)
    super(FC2Layer, self).__init__()
    self.input_size = input_size
    # 这里直接用 Sequential 就定义了网络，注意要和下面 CNN 的代码区分开
    self.network = nn.Sequential(
        nn.Linear(input_size, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, output_size),
        nn.LogSoftmax(dim=1)
    )
  def forward(self, x):
    # view一般出现在model类的forward函数中，用于改变输入或输出的形状
    # x.view(-1, self.input_size) 的意思是多维的数据展成二维
    # 代码指定二维数据的列数为 input_size=784，行数 -1 表示我们不想算，电脑会自己计算对应的数字
    # 在 DataLoader 部分，我们可以看到 batch_size 是64，所以得到 x 的行数是64
    # 可以加一行代码：print(x.cpu().numpy().shape)
    # 训练过程中，就会看到 (64, 784) 的输出，和我们的预期是一致的

    # forward 函数的作用是，指定网络的运行过程，这个全连接网络可能看不啥意义，
    # 下面的CNN网络可以看出 forward 的作用。
    x = x.view(-1, self.input_size)
    return self.network(x)

  class CNN(nn.Module):
      def __init__(self, input_size, n_feature, output_size):
          # 执行父类的构造函数，所有的网络都要这么写
          super(CNN, self).__init__()
          # 下面是网络里典型结构的一些定义，一般就是卷积和全连接
          # 池化、ReLU一类的不用在这里定义
          self.n_feature = n_feature
          self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_feature, kernel_size=5)
          self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
          self.fc1 = nn.Linear(n_feature * 4 * 4, 50)
          self.fc2 = nn.Linear(50, 10)

      # 下面的 forward 函数，定义了网络的结构，按照一定顺序，把上面构建的一些结构组织起来
      # 意思就是，conv1, conv2 等等的，可以多次重用
      def forward(self, x, verbose=False):
          x = self.conv1(x)
          x = F.relu(x)
          x = F.max_pool2d(x, kernel_size=2)
          x = self.conv2(x)
          x = F.relu(x)
          x = F.max_pool2d(x, kernel_size=2)
          x = x.view(-1, self.n_feature * 4 * 4)
          x = self.fc1(x)
          x = F.relu(x)
          x = self.fc2(x)
          x = F.log_softmax(x, dim=1)
          return x;

      # 测试函数
      def train(model):
          model.train()
          # 从train_loader里，64个样本一个batch为单位提取样本进行训练
          for batch_idx, (data, target) in enumerate(train_loader):
              data = data.to(device)
              target = target.to(device)
              optimizer.zero_grad()
              output = model(data)
              loss = F.nll_loss(output, target)
              loss.backward()
              optimizer.step()
              if batch_idx % 100 == 0:
                  print('Train:[{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(batch_idx * len(data), len(train_loader.dataset),
                                                                      100. * batch_idx / len(train_loader),
                                                                      loss.item()))

      def test(model):
          model.eval()
          test_loss = 0
          correct = 0
          for data, target in test_loader:
              # 把数据传入GPU中
              data, target = data.to(device), target.to(device)
              # 把数据送入模型，得到预测结果
              output = model(data)
              # 计算本次batch的损失，并加入到test_loss中
              '''
              output.max(1, keepdim=True)--->返回每一行中最大的元素并返回索引，返回了两个数组
              output.max(1, keepdim=True)[1] 就是取第二个数组，取索引数组。
              '''
              test_loss += F.nll_loss(output, target, reduction='sum').item()
              # get the index of the max log-probability, 最后一层输出10个数
              # 值最大的那个即对应着分类结果，然后把分类结果保存到pred里
              pred = output.data.max(1, keepdim=True)[1]
              # 将 pred 与 target 相比，得到正确预测结果的数量，并加到 correct 中
              # 这里需要注意一下 view_as ，意思是把 target 变成维度和 pred 一样的意思
              correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

          test_loss /= len(test_loader.dataset)
          accuracy = 100. * correct / len(test_loader.dataset)
          print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, len(test_loader.dataset),
              accuracy))
          n_hidden = 8
          model_fnn = FC2Layer(input_size, n_hidden, output_size)
          model_fnn.to(device)
          optimizer = optim.SGD(model_fnn.parameters(), lr=0.01, momentum=0.5)
          print('Number of parameters: {}'.format(get_n_params(model_fnn)))

          train(model_fnn)
          test(model_fnn)