---
title: "深度学习优化技巧：Dropout 与 Batch Normalization"
date: "2025-08-03T11:40:47+08:00"
draft: false
description: "深入解析深度学习中两大关键技术：Dropout正则化与Batch Normalization优化方法，包含核心原理、实现细节和实际应用指南。"
tags: 
    - dropout
    - batch-normalization
    - regularization
    - deep-learning
    - optimization
    - neural-networks
keywords:
    - dropout-regularization
    - batch-normalization
    - deep-learning-optimization
    - neural-network-training
    - overfitting-prevention
    - gradient-descent
series: "hylee"
---
这篇博文通过解释 Dropout 和 Batch Normalization 的核心原理、实现细节和主要作用，旨在帮助读者理解这两种在深度学习训练中常用的正则化与优化技术。

<!--more-->

## Dropout

Dropout 是神经网络训练过程中的一种常用技术，其核心思想是，在每次训练迭代的时候按照一定的给定概率 $p$，" 丢弃 " 全连接层中的部分神经元。这意味着，在每次前向传播的过程中，模型使用的网络是全连接网络的一个子集。从另一个角度上看，这就像在训练不同神经元排列组合的无数模型，再从中进行选取。这正是模型集成 (model ensemble) 的方法。

这从本质上讲，是在网络训练过程中，添加噪声以维持模型对于微小噪声的稳定性 [^1]。

在实际计算中，有两个细节需要注意

- 当按照概率 $p$ 舍去一部分节点的时候 - 即给这些节点在下面的计算中赋值 0, 为了维持该层输出的总期望在训练和测试时保持一致，要对未被丢弃的节点进行放缩，具体来说就是要将原输出值除以 $1-p$ 比如，当 $p=0.5$ 时，我们需要随机将一半的节点赋值为 0,同时对另一部分节点的数值翻倍 ($1-0.5$) 以保持期望不变 [^1]。
- 测试时不需要 dropout。使用框架的 dropout 层会自动实现，而无需额外注意。
 	- 我对这个原因的理解是，在训练过程中加入噪声扰动以增强模型的 robust，测试时使用的已经是增强的模型，直接使用以获得最好测试结果。可以想象，在测试时也使用 dropout 的话，测试结果会面临一定程度的下滑。

## Batch Normalization

### 名词辨析

在深入了解 Batch Normalization 之前，我们先厘清三个容易混淆的概念：

- 正则化 (Regularization)：这是一个**目标**。其目的是防止模型过拟合，提高其在未见过数据上的泛化能力。通常的手段是在训练过程中加入微小噪声，实现手段包括 L1/L2 惩罚项、**Dropout**，以及我们后面会看到的，**Batch Normalization** 带来的 " 副作用 "。
- 归一化 (Normalization)：这是一个宽泛的**数据预处理过程**，指将数据映射到特定范围。最常见的 " 归一化 " 特指 **Min-Max Scaling**，将数据缩放到 [0, 1] 或 [-1, 1] 区间。
- 标准化 (Standardization：这是归一化的一种**特定方法**，即 Z-score Normalization。它将数据处理成均值为 0，标准差为 1 的分布。**Batch Normalization** 和 Layer Normalization 本质上做的就是标准化，只是应用的维度和数据范围不同。

### 批量归一化

这个方法的主要目的，是把大的深的网络 train 起来，在比较少的迭代中收敛 [^2]。正则化是偶然的收获。

feature normalization - 可以改变 error surface 的形状，把崎岖变得平坦，方便优化搜索。这与自适应学习率、选用不同优化器 - 改变损失函数和优化方法，是两个优化方向 [^3]。

z 值归一化 - z score normalization 其中一种。计算同一维度分量的均值和标准差，也就是计算对应同一 feature 的不同样本表现的均值和标准差，利用它们对每个特征的数值进行归一化。我们得到的最终结果，在同一维度上变量的数值，符合均值 0 方差 1。

每次通过一层网络 - 线性回归之后，变量的分布和范围都可能发生变化，所以都要进行归一化。其放置的具体位置，目前的主流实现和推荐都是放在**线性变换之后，激活函数之前**。

> 主要原因是，BN 的目的之一是解决 " 内部协变量偏移 "（Internal Covariate Shift），即网络深层节点输入分布不稳定的问题。将 BN 放在激活函数（如 ReLU 或 Sigmoid）之前，可以直接控制送入激活函数的数据分布，使其保持在梯度较明显的区域（例如，对于 Sigmoid 函数，是 0 附近导数大），从而避免梯度消失，加速训练。

特征归一化 - 指的是，对全体训练集数据进行归一化。而批量归一化，是对整个批量里面的数据，计算特定维度对应的均值和标准差，然后对这个批量的数据进行归一化。所以，要批量数量较大才可以，最好批量具有足够代表性，这样进行批量归一化就近似于进行全体数据的特征归一化。

批量归一化之后，要接一个~~线性回归层~~仿射变换，引入两个需要学习的参数 $a,\gamma$ 目的是，改变批量归一化强行改变的分布。如果是均值 0 方差 1 的分布，很可能会对后续造成不良影响，所以需要使用线性层调整一下分布。在使用主流框架的时候，无需关心，都已经实现并封装。

测试时候，对训练过程中批量归一化得到的均值和标准差进行移动平均 $p\bar{\mu}-(1-p)\mu^{t}\to\bar{\mu}$，其参数 $p$ 也是个需要调的超参数，当作测试时进行归一化实际使用的参数。

最后，实验可以证明，batch norm 得到平坦的 error surface ，可以使用更大的 lr 进而快速的收敛。

总而言之，如果说 Dropout 像是在训练时让每个神经元 " 轮流休假 " 以增强团队合作能力，那么 Batch Normalization 就像是在每一层网络前都设置了一个 " 纪律委员 "，确保传递给下一层的数据 " 行为规范 "（分布稳定），让整个学习过程更加高效有序。

> 来自 LLM 的神奇比喻收尾

## Reference

[^1]:<https://d2l.ai/chapter_multilayer-perceptrons/dropout.html#dropout>
[^2]:<https://d2l.ai/chapter_convolutional-modern/batch-norm.html#batch-normalization>
[^3]:<https://github.com/datawhalechina/leedl-tutorial>
