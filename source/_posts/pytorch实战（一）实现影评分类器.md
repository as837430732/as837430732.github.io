---
title: pytorch实战（一）实现影评分类器
date: 2020-06-09 16:59:48
tags:   
 - [Pytorch]
categories:   
 - [深度学习,实战]      
kewords: "关键词1,关键词2"  
description: "使用pytorch实现一个二元影评分类器，数据使用IMDB"  
cover: "https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%B8%80/pytorch.jpg"
---

# Pytorch实现文本分类，数据集为kears内置IMDB影评数据

## 1.  导入相应库、定义常量以及加载IMDB数据

- 导入库，其中imdb数据从keras中导入

  ```python
  import torch
  import torch.nn as nn
  import torch.optim as optim
  import torch.nn.functional as F
  from torch.utils.data import *
  from keras.datasets import imdb
  from keras.preprocessing.sequence import pad_sequences
  ```

  

- 需要定义的常量：词汇表大小、句子最大长度、批处理量、嵌入层层数、隐藏层层数、设备

  ```python
  MAX_WORDS = 10000 #词汇表大小
  MAX_LEN = 200
  BATCH_SIZE = 256
  EMB_SIZE = 128
  HID_SIZE = 128
  DROPOUT = 0.2
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
  ```
  
- 加载IMDB数据，TensorDataset->RandomSampler->DataLoader

  ```python
  (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS) # 加载数据
  # 将训练集、测试集中的文本进行预处理，变成相同长度的文本，这里采用的规则是，在句子后面填充或截断
  x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
  x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')
  print(x_train.shape, x_test.shape)
  
  train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
  test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
  
  train_sampler = RandomSampler(train_data) # 从一个打乱的数据集进行采样
  train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE) # 将数据打包起来（一个batch_size是一组）
  
  test_sampler = RandomSampler(test_data)
  test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
  ```

## 2. 定义模型

- 使用LSTM模型进行文本分类，模型类中需要初始化函数、前向函数。

- 初始化函数负责初始化模型的参数（词汇表大小、批处理量、嵌入层层数、隐藏层层数）以及模型的架构（本实验使用的是LSTM+线性层1+线性层2）。

- 前向函数负责输出最终分类结果（本实验通过Embedding->dropout->LSTM->dropout->fc1->relu->avg_pool2d->fc2最终得到二分类结果）

  ```python
  class Model(nn.Module):
      def __init__(self, max_words, emb_size, hid_size, dropout):
          super(Model, self).__init__()
          self.max_words = max_words
          self.emb_size = emb_size
          self.hid_size = hid_size
          self.dropout = dropout
          self.Embedding = nn.Embedding(self.max_words, self.emb_size)
          self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                              batch_first=True, bidirectional=True) # 两层双向LSTM
          self.dp = nn.Dropout(self.dropout)
          self.fc1 = nn.Linear(self.hid_size*2, self.hid_size)
          self.fc2 = nn.Linear(self.hid_size, 2)
  
      def forward(self, x):
          x = self.Embedding(x) # x.shape [batch_size, max_len, emb_size]
          x = self.dp(x)
          x, _ = self.LSTM(x) # [batch_size, max_len, hid_size*2]
          x = self.dp(x)
          x = F.relu(self.fc1(x)) # [batch_size, max_len, hid_size]
          x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze() # [batch_size, 1, hid_size] -> [batch_size, hid_size]
          out = self.fc2(x) # [batch_size, 2]
  
          return out
  ```

## 3. 定义训练函数

- 训练函数中首先定义损失函数（本实验使用交叉熵），然后加载数据，并通过将数据输入到模型中产生分类结果（y_），将损失计算后，通过反向传播（loss.backward()）更新模型参数（需要加上optimizer.step()模型参数才会被更新）

  ```python
  def train(model, device, train_loader, optimizer, epoch):
      model.train() # 当有`Dropout`, `BatchNorm`时，需要加上这条
      criterion = nn.CrossEntropyLoss() # 交叉熵
      for batch_idx, (x,y) in enumerate(train_loader):
          x, y = x.to(device), y.to(device) # x 是一个二维tensor,其中每一行是一个句子，一组batch_size个句子
          y_ = model(x)
          loss = criterion(y_, y)
          loss.backward()
          optimizer.step()
  
          if(batch_idx + 1)%10 == 0:
              print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                  epoch, batch_idx * len(x), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()
              ))
  ```

## 4. 定义测试函数

- 测试函数与训练函数类似，而测试函数中需要计算准确率。

  ```python
  def test(model, device, test_loader):
      model.eval()
      criterion = nn.CrossEntropyLoss(reduction='sum')
      test_loss = 0.0
      acc = 0
      for batch_idx, (x, y ) in enumerate(test_loader):
          x, y = x.to(device), y.to(device)
          # torch.no_grad()，对tensor的操作正常进行，但是track不被记录，无法求其梯度
          with torch.no_grad():
              y_ = model(x)
          test_loss += criterion(y_, y)
          pred = y_.max(-1, keepdim=True)[1] # .max()的输出为最大值和最大值的index， 获取index
          acc += pred.eq(y.view_as(pred)).sum().item()
      test_loss /= len(test_loader.dataset)
      print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
          test_loss, acc, len(test_loader.dataset),
          100. * acc/len(test_loader.dataset)
      ))
      return acc/ len(test_loader.dataset)
  ```

  

## 5. 训练+保存模型

- 初始化模型，定义优化函数（本实验使用的Adam优化器），定义模型保存路径

  ```
  model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
  print(model)
  optimizer = optim.Adam(model.parameters())
  
  best_acc = 0.0
  PATH = './model.pth' # 模型保存路径
  
  # 训练、测试以及保存模型
  for epoch in range(1, 5):
      train(model,DEVICE,train_loader,optimizer,epoch)
      acc = test(model,DEVICE,test_loader)
      if best_acc < acc:
          best_acc = acc
          torch.save(model.state_dict(), PATH)
      print("acc is {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
  
  ```

  

## 6. 加载+测试模型

- 加载保存的模型，调用测试函数进行测试

  ```python
  best_model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT)
  best_model.load_state_dict(torch.load(PATH))
  test(best_model, DEVICE, test_loader)
  ```

  

## 7. 输出结果

- ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%B8%80/1.png)
- ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%B8%80/2.png)