---
title: Windows+Hexo+Github搭建个人博客
date: 2020-05-22 16:03:57  
tags:   
 - [Windows]
 - [Hexo]
 - [GitHub]  
categories:   
 - [教程]      
kewords: "关键词1,关键词2"  
description: "记录自己搭建博客的过程，帮助更多的人搭建属于自己的个人博客"  
cover: "https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/cover.jpg"

---

# Windows+Hexo+GitHub搭建个人博客

## 1. Git

### 1.1 Git的下载以及安装

* [Git下载链接](https://git-scm.com/download/win)

![Git下载](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Git下载.png)

* 下载完成后，双击可执行文件后，一直点击Next即可（全部默认选项）。

   ![Git1](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Git1.png)

   ![Git2](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Git2.png)

* 安装完成后，在桌面鼠标右键就会出现Git GUI Here和Git Bash Here，还可以通过在命令提示符（win+r）中输入`git` 来检测git是否已经成功安装。

   ![Git3](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Git3.png)

   ![Git4](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Git4.png)

### 1.2 Git的配置

* 申请[GitHub账号](https://github.com/)

* 在桌面鼠标右击后点击Git Bash Here，键入以下内容（请复制除#以外内容）

   ```bash
   # 配置用户名
   git config --global user.name "username"   
   #"username"是自己的账户名
   # 配置邮箱
   git config --global user.email "username@email.com"     
   #"username@email.com"注册账号时用的邮箱
   # 生成ssh
   ssh-keygen -t rsa 
   ```

* 在连续回车后，会出现RSA密码已生成的图样，到C:\Users\你的用户名.ssh目录下找到id_ras.pub文件，将该文件打开并复制其中内容（公钥）,将该内容粘贴到Github管理平台中的SSH and GPG keys（可在Github个人账户的设置中找到）,Title可以随便起，Key就是刚才复制的公钥。

   ![Git5](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Git5.png)

* 测试一下配置是否成功，在Git Bush命令框（就是刚才配置账号和邮箱的命令框）中继续输入以下命令。

   ```bash
   ssh -T git@github.com
   ```

   出现如下情况就说明已经配置成功

   ![Git6](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Git6.png)

## 2. NodeJs

### 2.1 NodeJs的下载以及安装

* [NodeJs下载链接](https://nodejs.org/en/download/)

![NodeJs下载](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/NodeJs下载.png)

* 下载完成后，双击可执行文件后，一直点击Next即可（其中安装路径要修改，不要安装在C盘）。

   ![NodeJs1](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/NodeJs1.png)

* 安装完成后，可以通过在命令提示符中输入node -v检查是否成功安装（如果出现版本号，说明已经成功安装）

### 2.2 NodeJs的配置

* 这里的配置很重要，原因是我们修改了NodeJs的安装路径，但是npm下载的模块还是会安装在默认的路径当中（也就是在安装NodeJs时的默认路径），后续在安装大量的模块后，会导致C盘中的内容过多，因此我们需要手动的修改npm安装模块的目录，在NodeJs文件夹下（我的安装路径是D:\NodeJs）新建两个文件夹，分别命名为node_global、node_cache

   ![NodeJs2](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/NodeJs2.png)

* 打开命令提示符，使用下面命令将npm的全局模块目录和缓存目录配置到我们刚才创建的这两个目录中

   ```bash
   npm config set prefix "node_global文件的路径"
   npm config set cache "node_cache的文件路径"
   ```

   我这里的文件路径对应的是“D:\NodeJs\node_global”和”D:\NodeJs\node_cache“

* 上述步骤完成后，需要在环境变量中添加路径（我这里的路径是”D:\NodeJs\node_global“，根据自己的安装路径来填写）

   ![NodeJs3](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/NodeJs3.png)

   ![NodeJs4](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/NodeJs4.png)

* 配置国内源，打开命令提示符输入如下内容即可

   ```bash
   npm --registry https://registry.npm.taobao.org
   ```

   新手可以参考这两个教程安装：

   1. https://jingyan.baidu.com/article/48b37f8dd141b41a646488bc.html
   2. https://blog.csdn.net/qq_43285335/article/details/90696126
   
   

## 3. Hexo

* 新建一个文件夹MyBlog（用来存放所有Blog的内容）

* 打开命令提示符，然后cd到该文件夹下输入如下命令，安装hexo

   ```bash
     npm i -g hexo
   ```

   ![Hexo1](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Hexo1.png)

* 安装完成后，可通过`hexo -v`查看版本，在MyBlog文件夹中右击后点击Git Bash Here，输入下述命令来初始化我们的博客（请注意，我们一定在刚才创建的文件夹路径下键入命令）

   ```bash
     hexo init
   ```

   ![Hexo3](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Hexo3.png)

   一段时间的等待后就可以看到MyBlog文件夹下出现了很多新的文件夹和文件
   ![Hexo2](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/Hexo2.png)

## 4. GitHub

* 按照图片中的步骤在GitHub中新建两个仓库，分别命名为：用户名.github.io，Blog_images，第一个仓库用于存储你的博客，第二个仓库用于存储你博客中的图片

   ![GitHub1](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/GitHub1.png)

   ![GitHub2](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/GitHub2.png)

   ![GitHub3](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/GitHub3.png)

## 5. 发布网站

* 首先将Hexo和GitHub进行关联，打开MyBlog文件夹下的_config.yml，找到deploy

   ![GitHub4](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/GitHub4.png)

   将deploy中的内容修改为以下内容

   ```bash
   deploy:
      type: git
      repo: https://github.com/你GItHub的账户名称/你GItHub的账户名称.github.io.git
      branch: master
   ```

* 安装Git部署插件，在Git Bash中输入如下内容

   ```bash
    npm install hexo-deployer-git --save
   ```

   ![GitHub5](https://raw.githubusercontent.com/as837430732/Blog_images/master/%E6%95%99%E7%A8%8B/GitHub5.png)
   再输入如下命令就可以将博客部署到GitHub中

   ```bash
    hexo clean     #清除缓存
    hexo g         #生成
    hexo d         #部署
   ```

   通过在浏览器中输入`https://你GItHub的账户名称.github.io/` 就可以访问你的个人博客了

## 6. 更换博客主题


   * 可以通过[主题](https://hexo.io/themes/) 选择自己喜欢的博客主题，不同的主题都有其对应的相关文档。

## 7. 对博客网站操作

### 7.1 新增博客

  ```bash
    hexo n "博客名字"         
  ```
### 7.2 删除博客
   * 在`source/_post`文件夹下，删除对应的md文件，然后通过`hexo g、hexo d` 生成、部署网页，即可成功删除。

### 7.3 调试博客
   * 通过命令`hexo s --debug` 在本地浏览器中输入`localhost:4000`可以预览博文效果。写好博文并且样式无误后，通过`hexo g、hexo d` 生成、部署网页，然后可以在浏览器中输入域名浏览博客。


## 参考文献

1. https://www.cnblogs.com/liuxianan/p/build-blog-website-by-hexo-github.html
2. https://www.cnblogs.com/du-hong/p/9921214.html
3. https://www.cnblogs.com/visugar/p/6821777.html
4. https://baidinghub.github.io/
5. https://blog.csdn.net/qq_43285335/article/details/90696126
6. https://blog.csdn.net/huangqqdy/article/details/83032408


