---
title: pytorch实战（二） 在MNIST数据集复现FGSM、DeepFool攻击
date: 2020-06-15 20:09:58
tags:   
 - [Pytorch]
categories:   
 - [深度学习,实战]      
kewords: "关键词1,关键词2"  
description: "使用pytorch实现对抗攻击"  
mathjax: true
cover: "https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%B8%80/pytorch.jpg"
---

# Pytorch实现FGSM、DeepFool攻击，数据集为MNIST

## 1 实现MNIST数据集分类

### 1.1 导入相应库、定义常量以及加载MNIST数据

- 导入库

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import  *
```

- 定义常量：批处理大小，设备

```python
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
```

- 利用torchvision加载MNIST数据集，图像数据一般需要自定义transform（data[0]\[0]表示第一张图片，data[1]\[1]表示第一张图片的标签）

```python
# 通过transform.Lambda来自定义修改策略
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Lambda(lambda  x : x.resize_(28*28))
                                      ])
traindata = torchvision.datasets.MNIST(root="./mnist", train=True,
                                       download=True, transform=mnist_transform)
testdata = torchvision.datasets.MNIST(root="./mnist", train=False,
                                      download=True, transform=mnist_transform)
# RandomSampler，当dataloader的shuffle参数为True时，系统会自动调用这个采样器，实现打乱数据。
train_loader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(testdata, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
```

### 1.2 定义模型

- 本实验使用的三层线性层，模型中需要定义初始化函数以及前向函数
- 线性层的结构定义跟图片维度、类别相关（本实验数据集为MNIST，一般处理为28*28）

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 1.3 定义训练函数

- 训练函数中需要定义损失函数，然后加载数据，并将数据输入至网络中，损失函数计算损失，将损失反向传播后，利用优化函数更新网络参数，需要注意一点（每一个mini-batch都需要将优化函数中的参数的梯度至零）。可以看出文本分类和图像分类的训练函数差别不大，网络的训练步骤是一样的。

```python
def train(model, device, train_loader, optimizer, epoch):
    loss_function = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
        loss = loss_function(y_, y)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            # print("x ", x)
            # print("batch_idx ", batch_idx)
            # print("len(x) ", len(x))
            # print("len(train_loader.dataset) ", len(train_loader.dataset))
            # print("len(train_loader) ", len(train_loader))
            # batch_idx:第几批次， len(train_loader.dataset):训练集大小， len(train_loader):总共有多少批次
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))
```

### 1.4 定义测试函数

- 图像分类的测试函数与文本分类并无太大差别，在这里就不详细介绍了。

```python
def test(model,device, test_loader):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0.0
    acc = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        # torch.no_grad()，对tensor的操作正常进行，但是track不被记录，无法求其梯度
        with torch.no_grad():
            y_ = model(x)
        test_loss += criterion(y_, y)
        # pred = y_.max(-1, keepdim=True)[1]  # .max()的输出为最大值和最大值的index， 获取index
        # acc += pred.eq(y.view_as(pred)).sum().item()
        _, pred = torch.max(y_ ,1)
        acc += (pred == y).sum().item() # type(acc) = python, 这里需要加上item()，不然type(acc) = tensor
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)
    ))

    return acc / len(test_loader.dataset)
```

### 1.5 训练+保存模型+加载模型+测试

- 初始化模型，优化器并定义模型保存路径

```python
model = Model()
print(model)
# 将网络参数放到优化器中
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)

best_acc = 0.0
PATH = './mnist_model.pth' # 模型保存路径
```

- 迭代训练，保存模型

```python
# 训练以及测试
for epoch in range(1, 5):
    train(model,DEVICE,train_loader,optimizer,epoch)
    acc = test(model,DEVICE,test_loader)
    print("acc ", acc)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), PATH)
    print("acc is {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
```

- 训练过程如下图所示

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%BA%8C/1.png)

- 加载模型，测试准确率

```python
best_model = Model()
best_model.load_state_dict(torch.load(PATH))
test(best_model, DEVICE, test_loader)
```

- 测试结果如下图所示

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%BA%8C/2.png)

## 2 实现FGSM攻击

### 2.1 导入相应库，定义常量，加载模型

- 需要定义的常量：模型保存路径，要扰动的图片，扰动程度（FGSM论文中的$\varepsilon$）

```python
import torch
import torch.nn as nn
from torch.autograd import Variable
from ImageClassification.MNISTClassification import Model,testdata
from torchvision import transforms
import matplotlib.pyplot as plt

PATH = './mnist_model.pth' # 模型保存路径
index = 100 # 选择测试样本
epsilon = 0.1 # 扰动程度

best_model = Model()
best_model.load_state_dict(torch.load(PATH))
```

### 2.2 实现FGSM

- 将图片输入到网络中，求出损失并反向传播
- 根据公式FGSM论文中的扰动表达式，$\varepsilon{sign(\nabla_{x}J(\boldsymbol{\theta},\boldsymbol{x},y))}$，求出扰动

- 需要注意：当需要用到tensor的梯度时，需要将数据定义为Variable，不明白可以参考

  https://www.jb51.net/article/177996.htm

  https://www.cnblogs.com/ryluo/p/10190218.html

```python
loss_function = nn.CrossEntropyLoss()
# 需要对image进行求导，因此转为Variable变量，并且将requires_grad设置为True
image = Variable(testdata[index][0].resize_(1,784), requires_grad=True)
label = torch.tensor([testdata[index][1]])
outputs = best_model(image)
loss = loss_function(outputs, label)
loss.backward()

# FGSM 添加扰动
x_grad = torch.sign(image.grad.data)
# torch.clamp(输入， 输入数据的最小值， 输入数据的最大值) 修剪，最大不超过1，最小不小于0
x_adversarial = torch.clamp(image.data + epsilon * x_grad, 0, 1)
```

- 测试攻击效果并展示攻击结果

```python
outputs = best_model(x_adversarial)
print(type(outputs))
predicted = torch.max(outputs.data, 1)[1] # image是Variable，因此outputs含有梯度值
print("original predicata is : {}".format(label[0]))
print("attacked predicate is : {}".format(predicted[0]))

print(x_adversarial)
x_adversarial.resize_(28, 28)
img_adv = transforms.ToPILImage()(x_adversarial)
plt.subplot(1, 2, 1)
plt.title("img_adv")
plt.imshow(img_adv)


img = testdata[index][0]
img = transforms.ToPILImage()(img.resize_(28, 28))
plt.subplot(1 , 2 , 2)
plt.title("img_org")
plt.imshow(img)

plt.show()
```

![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%BA%8C/3.png)

## 3 实现DeepFool攻击

### 3.1 导入相应库，定义常量，加载模型

- 需要定义的常量：模型保存路径，需要扰动的图片，迭代次数（DeepFool的扰动需要迭代添加），最大迭代次数，论文中的$\eta$（保证数据点跨越分界面）

  ```python
  import torch
  from ImageClassification.MNISTClassification import Model, testdata
from torch.autograd import Variable
  import numpy as np
  import copy
  from torch.autograd.gradcheck import zero_gradients
  from torchvision import transforms
  import matplotlib.pyplot as plt
  
  PATH = './mnist_model.pth'  # 模型保存路径
  index = 100
  loop_i = 0
  max_iter = 50
  overshoot = 0.02  
  
  net = Model()
  net.load_state_dict(torch.load(PATH))
  ```

### 3.2 实现DeepFool

- 获取原始图像的标签，实验中直接利用模型的前向函数得到标签，也可以直接将图片输入至网络中得到标签

  ```python
  # 得到原始图片类别
  image = Variable(testdata[index][0].resize_(1, 784), requires_grad=True)
  label = torch.tensor([testdata[index][1]])
  # flatten() 将多维数组转换为一维数组
  f_image = net.forward(image).data.numpy().flatten()  # shape [1, 10] -> [10]
  I = (np.array(f_image)).flatten().argsort()[::-1]  # [::-1]从后往前取数
  label = I[0]
  ```

  

- 初始化网络参数，其中包括：公式中需要使用的权重，扰动，分类器输出$w$,$r$,$f$

  ```python
  # 初始化网络参数
  input_shape = image.data.numpy().shape
  pert_image = copy.deepcopy(image)  # 深复制，复制前后两个对象单独存在，不互相影响
  w = np.zeros(input_shape)
  r_tot = np.zeros(input_shape)
  
  x = Variable(pert_image, requires_grad=True)
  fs = net.forward(x)  # shape [1, 10]
  # fs_list = [fs[0][I[k]] for k in range(len(I))] # 按照概率从大到小的顺序将fs值进行排序
  k_i = label
  ```

  

- 参照论文中的伪代码，写出迭代攻击算法，其中外循环控制扰动是否成功以及迭代次数是否超过最大限制，内循环控制找出图像点和k个分界面中距离最近的分界面，找到最近分界面后就可以求出需要扰动的距离。

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%BA%8C/4.png)

  ```python
  while k_i == label and loop_i < max_iter:
      pert = np.inf
      fs[0][I[0]].backward(retain_graph=True)  # 多次反向传播（多层监督）时，梯度是累加的
      orig_grad = x.grad.data.numpy().copy()  # 赋值，等号左右占用同一个地址
      # 从k个扰动距离中选择最近的距离
      for k in range(len(I)):
          zero_gradients(x)
          fs[0][I[k]].backward(retain_graph=True)
          cur_grad = x.grad.data.numpy().copy()
  
          w_k = cur_grad - orig_grad  # shape [1, 784]
          f_k = (fs[0][I[k]] - fs[0][I[0]]).data.numpy()  # shape [1]
  
          pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
          if pert_k < pert:
              pert = pert_k
              w = w_k
      r_i = (pert + 1e-4) * w / np.linalg.norm(w)
      r_tot = np.float32(r_tot + r_i)  # 由于决策面是非线性的，因此需要叠加扰动
  
      pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
      # x = Variable(pert_image, requires_grad=True)
      fs = net.forward(pert_image)
      k_i = np.argmax(fs.data.numpy().flatten())  # 扰动后的类别
      loop_i += 1
  
  r_tot += (1 + overshoot) * r_tot
  ```

  

- 测试攻击效果并展示攻击结果（代码与FGSM基本一致）

  ```python
  outputs = net(pert_image.data.resize_(1, 784))
  predicted = torch.max(outputs.data, 1)[1]
  print("original predicata is : {}".format(label))
  print("attacked predicate is : {}".format(predicted[0]))
  
  pert_image = pert_image.data
  pert_image = pert_image.resize(28, 28)
  img_adv = transforms.ToPILImage()(pert_image)
  plt.subplot(1, 2, 1)
  plt.title("img_adv")
  plt.imshow(img_adv)
  
  img = testdata[index][0]
  img = transforms.ToPILImage()(img.resize_(28, 28))
  plt.subplot(1, 2, 2)
  plt.title("img_org")
  plt.imshow(img)
  
  plt.show()
  ```

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%BA%8C/5.png)

## 4 参考文献

1.     https://github.com/chaoge123456/MLsecurity/blob/master/blog/FGSM%20and%20DeepFool/adversary_example.ipynb
2. https://blog.csdn.net/u011630575/article/details/78604226
3. https://blog.csdn.net/xiaoxifei/article/details/87797935
4. https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.html
5. https://arxiv.org/abs/1412.6572

## 附录

- 知识点1：copy()、deepcopy()与赋值的区别

  deepcopy() ，深复制，**即将被复制对象完全再复制一遍作为独立的新个体单独存在。**所以改变原有被复制对象不会对已经复制出来的新对象产生影响。（复制前后的两个对象占用不同的地址）

  = ，赋值，**并不会产生一个独立的对象单独存在，他只是将原有的数据块打上一个新标签**，所以当其中一个标签被改变的时候，数据块就会发生变化，另一个标签也会随之改变。（是同一个对象，占用相同的地址，只是名称不同）

  copy() ，浅复制，分为两种情况
  
  1）当浅复制的值是不可变对象（数值，字符串，元组）时和“等于赋值”的情况一样，对象的id值与浅复制原来的值相同。
  
  2）当浅复制的值是可变对象（列表和元组）时会产生一个“是那么独立的对象”存在。有两种情况：
  
  第一种情况：复制的 对象中无 复杂 子对象，原来值的改变并不会影响浅复制的值，同时浅复制的值改变也并不会影响原来的值。原来值的id值与浅复制原来的值不同。
  
  第二种情况复制的对象中有 复杂 子对象 （例如列表中的一个子元素是一个列表），如果不改变其中复杂子对象，浅复制的值改变并不会影响原来的值。但是改变原来的值 中的复杂子对象的值会影响浅复制的值。

- 知识点2：optimizer.step() 和loss.backward()

  ```python
  optimizer = Adam(model.parameters(), lr=0.01)
  loss_function =  nn.CrossEntropyLoss()
  
  for mini-batch:
      optimizer.zero_grad()
      根据输入求出loss
      loss.backward()
      optimizer.step()
  ```

  一般情况下，模型的训练代码都是这种结构的，优化器的作用是优化模型的参数，因此初始化时需要将模型的参数交给优化。优化器还需要损失反向传播的梯度信息，因此在loss.backward()之后接上optimizer.step()。切记，每一个mini-batch都需要将优化器中的梯度信息清零，以免影响后续的优化。

- 保存网络与加载网络

  ```python
  # 保存整个网络
  torch.save(net, 'mnist_net_all.pkl')
  # 保存网络中的参数, 速度快，占空间少
  torch.save(net.state_dict(),'mnist_net_param.pkl')
  # 针对上面一般的保存方法，加载的方法分别是：
  
  model_dict=torch.load(PATH)
  model_dict=model.load_state_dict(torch.load(PATH))
  ```

  