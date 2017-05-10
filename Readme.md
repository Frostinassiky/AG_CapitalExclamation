# [ AlphaGo！] 模型：ForkNet
by _Frost_, _Yiye_, and _Baojia_!

声明，该工作继承自 _endernewton_ 的faster-rcnn框架，原来的链接可以参考 https://github.com/endernewton/tf-faster-rcnn

## 算法介绍
该网络模型从faster-rcnn中进行改进，利用叉子状网络实现对性别的分类，和对帽子、口罩、以及眼镜的分类。

### 网络的建立
faster-rcnn网络是一个经典的利用proposal和classification相结合实现物体识别的深度神经网络模型，其最大的特点就是在深度的网络层次上提取物体的位置（bbox）特征，并在深度层直接进行感兴趣区域的提取，和bbox的精确定位。

深度学习网络结构如下图所示
![模型结构](https://raw.githubusercontent.com/Frostinassiky/AG_CapitalExclamation/AlphaNext/img/Slide2.JPG)
整体网络成叉子的形状，叉子手柄处为共用层，用于从图片中提取低级特征．大部分参数集中在多层全卷积网络部分，他们将一个(1x112x96x3)的图片抽象为（）的特征层．

叉子的最左边与传统的faster-rcnn一样，计算可疑区域(proposals)后再分类,进一步确定框的位置．
proposals 用的是传统的ＲＰＮ网络．

叉子的右边三支分别用于计算帽子\眼镜和口罩的分类．为了准确度，这三支同样加入了区域池化，在特征层当中截取可疑的区域进行计算．
颜色判断的可疑区域不通过ＲＰＮ提供，而是根据人头的检测结果，自主设计了proposals的分布．该设计如下图所示，由于不需要给出这些物体的位置，我们采用了SSD类似的方式直接给出proposals

![可疑区域设计](https://raw.githubusercontent.com/Frostinassiky/AG_CapitalExclamation/AlphaNext/img/Presentation1.jpg)

因此，使用该网络计算时相当于传播了两次，第一次得到性别信息和人头位置，第二轮根据人头位置和特征层对帽子眼镜口罩进行分类．

### 模型的训练
直接用后向传播的方法计算梯度不适用于该模型．这是因为RPN提供的proposals本身就会有一定不准确性，带来第一轮预测的误差．我们这里参考了四步训练法的简化训练方法，利用真实框的位置进行训练．训练中的模型如下图

![训练模型结构](https://raw.githubusercontent.com/Frostinassiky/AG_CapitalExclamation/AlphaNext/img/Slide1.JPG)

在整理数据的过程中，我们已经得到了真实框的位置，所以该叉状网络的训练可以同步进行．与之前不同，对剩余的三个分叉，由头的真实位置直接提供可疑区域的初值，进而在初值上通过SSD的方法计算proposals,而后分类．

*经过我们20多天的努力，模型的结构建立完成，但尚未完成剩余三个分叉的反向传播梯度，故无法进行整个模型的训练，十分可惜．好在模型的主干部分（第一个分叉）可以跑通，所以也决定上交。* 在tensorboard中浏览到的模型主体部分如下图所示．[图片细节](/img/fork.png)中添加了共享的参数和链接．
![实际模型（tensorflow）](https://raw.githubusercontent.com/Frostinassiky/AG_CapitalExclamation/AlphaNext/img/fork_main.png)

## 训练数据
这部分的描述请参看Yiye的文件：
## 模型参数
训练20000次后的模型参数储存在：
https://drive.google.com/open?id=0B9Ti5uHc-pQ4Y2QyTURKRWZqQmc

由于只完成了主干的训练，我们还**无法**给出完整的模型参数。**:(　:( :(**

## 模型输入
在训练阶段，模型的输入有三部分，分别是图片、头部的位置信息、帽子眼睛口罩的分类结果。

在测试阶段，模型的输入为一张图片。

1. 图片信息通过需调整为112\*96大小。 详细见_Preprocessing\ProduceFace.m_
2. 头部的位置信息是一个四维向量，格式是 _[x1,y1,x2,y2]_. 由于在WIDER FACE数据库中存放有头部位置，按照裁剪过程中的几何关系可以计算得出
3. 帽子眼睛口罩的分类分别为三个0-13的整数，每个数字代表的含义在_Preprocessing\MergeInfo.m_中提及。
4. 位置信息和分类信息整理后存为COCO数据库的格式，整理的代码为 Preprocessing/WriteJsonBbox.py，存放文件是https://drive.google.com/open?id=0B9Ti5uHc-pQ4cFY2SGpjUl9wTDA


## 模型输出
网络模型的输出是位置信息和分类信息，与训练过程中的模型输入相同。

## 算法检验
### 安装说明
本模型基于tensorflow框架，使用CUDA加速，并使用COCO数据集的API进行结果评估。在Ubuntu14中可以正常运行。

安装步骤如下。
1. 复制  整个项目到本地
```Shell
  git clone https://github.com/Frostinassiky/AG_CapitalExclamation.git
  ```

2. 编译 Cython 模块，这是为了使用原faster-rcnn框架中的一些函数
```Shell
  make clean
  make
  cd ..
  ```

4. 安装 [Python COCO API](https://github.com/pdollar/coco). 
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..，
 ```

### 程序测试
把测试文件夹命名为 _alpha2017_

放入 _data/coco/images/_ 文件夹下
在原来的目录下运行
  ```Shell
  python tools/test.py --model res101_faster_rcnn_iter_20000.ckpt --imdb coco_style_face
   ```
### 程序输出
```Shell
~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~
49.6
39.5
53.9
55.5
~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.909
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.444
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Wrote COCO eval results to: /home/xum/Documents/Git/AlphaNext/AlphaModel/AG_CapitalExclamation/output/default/coco_2017_alpha/default/res101_faster_rcnn_iter_20000/detection_results.pkl
 ```
## 参考

[1] Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[C]//Advances in neural information processing systems. 2015: 91-99.

[2] Chen X, Gupta A. An implementation of faster rcnn with study for region sampling[J]. arXiv preprint arXiv:1702.02138, 2017.

[3] Yang S, Luo P, Loy C C, et al. Wider face: A face detection benchmark[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 5525-5533.

[4] Lin T Y, Maire M, Belongie S, et al. Microsoft coco: Common objects in context[C]//European Conference on Computer Vision. Springer International Publishing, 2014: 740-755.

[5] Liu W, Anguelov D, Erhan D, et al. SSD: Single shot multibox detector[C]//European Conference on Computer Vision. Springer International Publishing, 2016: 21-37.

