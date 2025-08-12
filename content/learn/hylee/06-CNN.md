---
title: "从神经元到滤波器：两个故事下的卷积神经网络 (CNN)"
date: "2025-08-12T16:35:58+08:00"
draft: false
description: "通过神经元优化和滤波器检测两种视角，深入解析CNN的三大核心设计：局部感受野、参数共享和池化，理解其在图像处理中的优势与应用"
tags:
    - convolutional-neural-network
    - deep-learning
    - computer-vision
    - receptive-field
    - parameter-sharing
    - pooling
    - feature-extraction
keywords:
    - cnn-architecture
    - convolution-layer
    - feature-map
    - image-classification
    - neural-network-optimization
    - deep-learning-fundamentals
series: "hylee"
---

这篇博文通过两种不同的叙事角度，生动地解释了卷积神经网络（CNN）如何通过局部感受野、参数共享和池化这三大核心设计，来有效处理图像数据并解决传统全连接网络参数过多的问题。主要内容和细节多参考自 Hylee 的机器学习 2021 版本 [^1]。

CNN 是神经网络架构的一种，在图像处理领域所向披靡，也一般是深度学习教程中最先学习的网络架构。

图片通过模型，得到 $\hat{y}$，这是个 one-hot vector，其具体维度就对应着模型可以进行多少种分类，识别多少种图像。这本质上是分类问题，所以也用 Cross Entropy 来定义和计算模型的 loss。

图像尺寸以 - 100 x 100 x 3 为例子；长 100 pixel ，宽 100 pixel， 3 通道 RGB，如果数据集中的图片尺寸不同，可能需要预处理得到同一尺寸再送入模型进行训练。

如果使用 MLP，100 x 100 x 3 的 tensor(张量)，在输入模型的时候，要拉直成一个巨大的向量。这时候，输入向量的长度为 3 x 100 x 100, 如果进而通过 100 个神经元，全连接的情况下就已经 3 x 100^6 个参数，这还只是一层网络。模型容量太大，对于训练的硬件和技术都提出了极高的要求，有没有可能减少参数？把三维 tensor 拉平成一个向量，损失了空间结构的信息 - 在图片中，相近的像素点之间可能是有相关性的，如何尽可能保存相近像素的位置结构？

## receptive filed 和 parameter sharing 的故事

这就是 CNN 对与 MLP 根据图像的特点做出的主要改进，主要有两点 - 局部感受野和参数共享。

首先，第一层神经元不需要全连接到所有的输入，而只是看图片的一小部分。这很像人眼进行观察，通过动物的不同位置特点，来判断这是个什么动物。实际上，就是选择特定尺寸的 receptive field - 3 x 3 x 3 的一小块图像，按照特定 stride 进行移动采样，把小块图给到神经元。不同 receptive field 可以重叠，不要留空，容易错过特征。

接着是参数共享，不同位置的 receptive field 享受同一组参数，用于检验某一种 pattern。我们需要把整张图片，按照一定尺寸采样和移动，逐个送入，以同样的参数检验其是否符合某种 pattern。

假设我们有 64 个神经元，也就意味着我们可以检验 64 中 pattern。与拥有 64 个神经元的全连接网络，每个神经元都要接受整张图片的所有像素不同，CNN 的思想是一个神经元检验一个特定像素范围内的模式。这就既可以把整张图片都送入神经元观察，又减少了需要训练的参数。

假设我们在 3 x 3 x 3 的感受野，对于一个共享参数的神经元，只需要估计 27 + 1 - bias = 28 个参数，随后我们只要接着送入不同位置的感受野并调整参数即可。这样，一个神经元需要训练的参数，从全连接下一个神经元 3 x 100 x 100 + 1 个需要估计的参数减少到了 28 个。

这里的假设是，同一种有效的 pattern 可能出现在图片的不同位置。由于输入不同，共享参数的神经元对应的输出也不同。

讲道理，因为降低了需要估计的参数这会降低模型拟合能力，引入 model bias，但这种简化对于图像问题是完全可以接受的，转化到其他问题可不一定。

receptive field + share parameter，这叫作一个 convolution layer.

## filter 的故事

刚刚我们从 " 神经元如何偷懒 " 的角度理解了卷积层，现在我们换一个更主流、更工程化的视角——" 滤波器（Filter）" 的故事。其实，这两种说法描述的是同一件事。

我们有 convolution layer - 由一系列 filter 组成，每个 filter 都是 3 x 3 x # of channel，目的是检验一种特定的 pattern。

一个可能的 patter 例如，是不是这块只有对角线有数值,其余位置都是 0？这个 filter 的参数可能就是除了对角线为 1,其余都是 0 。我们把这个 filter 按照一定的 stride 移动采样整张图片，其输出的就是输入与这种 pattern 的接近程度。如果输入对角线都是 0, 那输出一定是 0。每个 filter 对应一组参数，我们的网络就在学习这一系列 filter 的参数。

假如我们的图片为 6 x 6, filter 为 3 x 3, stride 为 1, 那我们 convolution layer 的输出，就会是 4 x 4 x # of filter 的一张 feature map。我们把这个输出再看作一张图片，就是 长 4,宽 4, 通道数为 # of filter. 假如有 64 个 filters，他就是 4 x 4 x 64 的图片，再接入 convolution layer 继续处理。

> 这是不填充 - padding（valid padding） 的情况，虽然更通常情况下，我们都需要 padding 以保证输入和输出的尺寸是相同的（same padding）。对于一个尺寸为 F x F 的 filter，利用 $\text{padding}=\frac{F-1}{2}$ 。
>
> > 如果不 padding，保持 "valid" padding ，这会导致随着 convolution layer 的加深，网络的输入越来越小，信息损失。而利用 "same"padding 保持输入和输出为相同尺寸，这是有利于训练更深的 convolution layer。

假如我们构造多个 convolution layer，都是 3 x 3 的 filter，随着它们的叠加，filter 看到的图片范围也是逐渐增大的。因为第二层的 3 x 3 的采样中，每个被采样的数据点都代表原始图片的 3 x 3。也就是说，在 stride 为 1 的时候，第二层的 3 x 3 实际上能看原图中的 5 x 5。

这个版本中的 filter 就是上文中，共享参数的神经元。还有一些术语让我们进行统一，filter 的维度被称为 - kernel size，比如 3 x 3；feature map - 就是 convolution 的结果，即一个 filter 扫过整张图片产生的结果矩阵，其每个元素代表着对应 receptive filed 与特定检验 pattern 的契合程度。

## 池化

第三个设计，就是 polling - 池化。

就是 sub-sampling，不需要 train 参数，因为是固定方法采样，可以降低 feature map 的维度。

典型的方法是 max-polling, 划定尺寸为 2 x 2, 就是 feature map 中，不重叠的划分子块，每个子块用其中的 max value 进行代表。经过池化， 4 x 4 变为 2 x 2，进一步缩小了图片的尺寸。

这一设计的出发点是，图像中，我们抽调某些行和列，实际上并不妨碍整个图片的表达。除了降低尺寸之外，polling 会增强系统的 robust。feature map 是整张图像对于某一种 pattern 的强度检测，例如，是否呈现眼睛形状。当使用 max polling 进行信息提取，从图像特征来说，就是保留了一定范围内的最强表现。可以看作是，对于最强特征的位置偏移。

比如我们检测到**靠近左上角**的位置，有个眼睛形状的最大值，经过 max polling 之后，可能这个 value 在 feature map 上就变成了**左上角**，这就把这个眼睛形状的位置进行了偏移。这种对特征位置的 " 不敏感 "，在机器学习中被称为**平移不变性（Translation Invariance）**。这意味着，无论眼睛在图像的左上角还是稍微偏右一点，模型都能稳定地识别出它。从为模型添加微小噪声可以增强模型 robust 的角度来说，这种位置的改变，是有利于模型的泛化的。

polling 总是会损失信息的，这个技术主要是为了在有限计算资源下加速计算。目前计算资源比较强大的情况下，也很**有多工作不再使用 polling**，直接大力出奇迹。

一个常规的 CNN 架构，从 CONV -> RELU -> POOL ，如此循环。接上一个 flatten layer 展平结果，再接上一个 fully connected layer + softmax，就得到了代表分类结果的 one-hot vector。

> 关于池化，有争论在于其过度激进地丢弃了大多数信息，也有 avg 代替 max 进行 polling 的方法。 同样可以考虑，**使用带 stride 的 CONV 以 sub-sampling 的方式代替池化**。

> 后来有 global polling，如 Global Average Pooling & Global Max Pooling。每个 feature-map 只输入一个 value，把 4 x 4 x 3 直接变成三维的向量，每个分量代表一个 feature map 的 avg 或者 map。用这种方式代替传统的 flatten + fully connected，可以减少需要估计的参数，还能增强模型的解释性。

## CNN 与 Alpha GO

CNN 设计影像问题专用，所以泛化到其他类型的问题时候，需要考虑问题本身是否与 CNN 的设计理念相吻合。

Alpha GO 就是 CNN 架构的结果。我们把围棋棋盘看作一张图片，黑子白字和无子对应不同数字编号；每个点对应的不同可能状态对应不同通道，比如，是否下一步会被吃；把下一步落子的位置，看作是 19 x 19 分类的结果。

> 热知识，围棋是 19 x 19 的棋盘。

为什么可以使用 CNN？围棋运动本身，就是一小块一小块状态的叠加，同一状态可能出现在棋盘中的不同位置。 值得注意的是，alpha go 就没用池化，可能对与围棋问题，sub-sampling 会导致完全不同的结果。

CNN 其实，对放大、缩小和旋转的图片都无能为力，不具有这种泛化。在训练中，通常在把图片送入模型之前，会对图片做随机的数据增强，以来增强 CNN 的能力。这样的用意在于使得模型学到的是一类物品，而不是一个完全确定的图片。

Vision Transformer (ViT) 从架构上进行改变，把图像看作类似 word sequence 传入 Transformer，在大规模数据和更大程度的数据增强上取得了优秀的成绩。

## Reference

[^1]:<https://www.youtube.com/watch?v=OP5HcXJg2Aw>
