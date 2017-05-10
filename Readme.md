# [ AlphaGo！] 模型：ForkNet
by _Frost_, _Yiye_, and _Baojia_!

声明，该工作继承自_endernewton_的faster-rcnn框架，原来的链接可以参考 https://github.com/endernewton/tf-faster-rcnn

## 算法介绍
该网络模型从faster-rcnn中进行改进，利用叉子状网络实现对性别的分类，和对帽子、口罩、以及眼镜的分类。

### 网络的建立
faster-rcnn网络是一个经典的利用proposal和classification相结合实现物体识别的深度神经网络模型，其最大的特点就是在深度的网络层次上提取物体的位置（bbox）特征，并在深度层直接进行感兴趣区域的提取，和bbox的精确定位。

 TODO

## 训练数据
这部分的描述请参看文件：
## 模型参数
训练20000次后的模型参数储存在：
https://drive.google.com/open?id=0B9Ti5uHc-pQ4Y2QyTURKRWZqQmc


由于只完成了主干的训练，我们还无法给出完整的模型参数。 :(

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
