---
title: pytorch基础（一）数据读取DataLoader和Dataset
date: 2020-11-03 16:59:48
tags:   
 - [Pytorch]
categories:   
 - [深度学习,基础]      
kewords: "关键词1,关键词2"  
description: "详细描述DataLoader读取数据机制"  
cover: "https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%AE%9E%E6%88%98/%E4%B8%80/pytorch.jpg"

---

# Pytorch数据读取DataLoader和Dataset

## 1.  DataLoader

DataLoader包括Sampler（用于生成索引）和Dataset（根据索引读取图片，标签）

![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/1.png)

DataLoader用于构建可迭代的数据装载器

```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
```

![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/2.png)

## 2. Dataset

Dataset是抽象类，所有自定义的Dataset均需要继承该类，并且重写\__getitem__()方法。\__getitem()__方法的作用是接收一个索引，返回索引对应的样本和标签，**这是我们自己需要实现的逻辑**。

![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/3.png)

## 3. 读取数据过程

首先在for循环中使用DataLoader，根据是否需要多进程读取数据选择DataLoaderIter，然后通过Sampler读取index列表，在DatasetFetcher中调用Dataset的getitem方法，根据索引index从硬盘读取（图片img，标签label）列表，再通过collate_fn将列表分为batch，batch由图片img列表和标签label列表组成。

![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/4.png)

详细过程如下描述：

- 首先进入for循环

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/5.png)

- 判断单进程还是多进程，程序中使用的单进程，因此进入_SingleProcessDataLoaderIter

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/6.png)

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/7.png)

- 初始化\_SingleProcessDataLoaderIter类之后，跳转到了DataLoader的\__next__()方法。

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/8.png)

- 跳转到\_SingleProcessDataLoaderIter类的\_next_data()方法，其中\_next_index()方法会调用Sampler类的\__iter__()方法，用来读取index列表。

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/9.png)

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/10.png)

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/11.png)

- 将index列表输入到DatasetFetcher类中，通过dataset[idx]获取图片和标签，dataset[idx]会调用Dataset的\__getitem__()方法。我们在RMBDataset中已经写好了从硬盘读取图片和标签的代码。data是由一个batch的（图片，标签）列表组成。

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/12.png)

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/13.png)

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/14.png)

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/15.png)

- data在经过collate_fn处理后，转变为两个列表，分别是图片列表和标签列表

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/16.png)

  ![](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/pytorch%E5%9F%BA%E7%A1%80/%E4%B8%80/17.png)