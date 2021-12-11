# 配置数据集

## 分配方式

数据集一般可以划分为训练数据集、验证数据集和测试数据集，对此有两种分配方式：

1. 训练数据集+验证数据集+测试数据集

   先用训练集训练出一组参数w和b，然后用验证集来评估这组参数的表现，如果表现不好就调整超参数，再用训练集来训练出一组新的参数，然后再用新的参数来使用验证集进行评估，如果不好就再调整超参数，再训练，再评估，直到结果满意，最终再用测试集进行参数评估。若测试集的评估结果也不好，那么又得重新开始训练，从这个角度来看，测试集就变成了验证集。

2. 训练数据集+测试数据集

   此处测试数据集的功能其实就等同于第一种方式的验证数据集。

## 分配比例

1. 对于小型数据集，一般分配比例是70/30（如果只分训练集/测试集）或60/20/20（如果分训练集/验证集/测试集）。
2. 对于大型数据集，一般分配比例是98/1/1或995/4/1。因为验证集和测试集的目的只是评估而已，对于海量数据集来说，1有可能就有一千多条数据。

## 划分原则

1. 尽量保证训练数据集、验证数据集和测试数据集的来源一致。
2. 请勿对测试数据进行训练。

# 拟合度

拟合度是指训练好的模型与训练数据集的拟合程度。

1. 欠拟合：对训练数据集预测的不准确。
2. 过拟合：对训练数据集的预测准确度非常高，但对验证数据集的预测准确度却很低。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514003136231.png" alt="image-20210514003136231" style="zoom:67%;" /> 

## 拟合度的判断

假设识别猫产生四组预测的错误率：

```
					 第一组		第二组		第三组		第四组
对训练集预测错误率		15%			 0.5%		1%		  15%
对测试集预测错误率		16%			  1%	    11%		  30%
```

1. 第一组属于欠拟合：对训练集预测错误率太高。
2. 第二组属于拟合刚好：对训练集和测试集的预测错误率都与人相似。
3. 第三组属于过拟合：对训练集预测错误率低，但对测试集预测错误率高，两者相差太远。
4. 第四组属于既欠拟合又过拟合：对训练集预测错误率高，且对测试集预测错误率更高，两者相差也太远。

## 解决欠拟合

1. 尝试更大的神经网络（增加神经网络的层数、增加神经元的个数）。
2. 增加训练次数（有可能解决欠拟合，但解决不了也没有害处，只是多花了些时间）。
3. 尝试其他的优化算法。
4. 尝试不同的神经网络架构也有可能解决欠拟合。

## 解决过拟合

1. 最佳方法就是获得更多的训练数据，但是有时候我们无法获得更多的训练数据，即使能获取也太贵了。
2. 使用正则化来解决过拟合的问题。
3. 尝试不同的神经网络架构也有可能解决过拟合。也就是说，如果找到一个合适的架构，那么可以同时解决欠拟合与过拟合的问题。
4. 提前终止迭代：在神经网络的训练过程中，初始化一组较小的权值参数，此时模型的拟合能力较弱，通过迭代训练来提高模型的拟合能力，随着迭代次数的增大，部分的权值也会不断的增大。如果我们提前终止迭代可以有效的控制权值参数的大小，从而降低模型的复杂度。
5. 权值共享：权值共享的目的旨在减小模型中的参数，同时还能较少计算量。
6. 增加噪声：这也是深度学习中的一种避免过拟合的方法。添加噪声的途径有很多，可以在输入数据上添加，增大数据的多样性，也可以在权值上添加噪声，这种方法类似于正则化。

## 对过拟合的理解

过拟合就是指机器模型过于自信，已经到了自负的阶段。自负的坏处就是在自己的小圈子里表现非凡，但在现实的大圈子里却往往处处碰壁。过拟合的根本原因就是机器模型急于想要减少误差，导致它拟合的曲线几乎经过了每一个数据点。

解决过拟合的方法：

1. 增大数据集。如果数据增多，则模型就不会过分依赖于某一个数据点（通常是错误的数据点）。

2. L1、L2正则化。假设W为机器需要学习到的各种参数。在过拟合中，W的值往往变化得特别大或特别小，为了不让W变化太大，我们就可以在计算误差上做些手脚。原始的误差是这样计算，$cost = (预测值-真实值)^2$，如果W变得太大，我们就让cost也跟着变大，作为一种惩罚机制。因此，我们把W自己考虑到误差中，使得机器不会过分依赖于某个参数，这就是L1、L2正则化的做法。

   ![overfitting5.png](https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/overfitting5.png)

3. dropout正则化。通过随机drop掉某些神经元的规则，我们可以想象其实每次训练的时候，让每一次预测结果都不会依赖于其中某部分特定的神经元，就像L1、L2正则化一样，如果机器过度依赖W，也就是训练参数的数值会很大，L1、L2正则化就会惩罚这些大的参数。Dropout的做法是从根本上让神经网络没机会形成过度依赖。

   <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/overfitting6.png" alt="overfitting6.png" style="zoom:67%;" />

# 正则化

正则化就是给需要训练的目标函数加上一些规则或限制，使他们不会自我膨胀，而变得过于复杂。

根据奥卡姆剃刀定律，我们可以通过**降低模型的复杂度**来防止过拟合，这种原则也就是正则。也就是说，加入正则后训练模型就不是以最小化损失为目标：minimize(Loss(DatalModeI))，而是以最小化损失和复杂度为目标：minimize(Loss(DatalModeI) + complexity(ModeI))。于是训练优化算法变成由两项内容组成的函数：一个是损失项，用于衡量模型与数据的拟合度；另一个是正则化项，用于衡量模型复杂度。

## L2正则化

> L1正则化也和L2正则化一样，其实现也是在函数后面加点尾巴。
>
> L1正则化和L2正则化的区别：L1使权重稀疏，L2使权重平滑，即L1会趋向于产生少量的特征，而其他的特征都是0；而L2会选择更多的特征，这些特征都会接近于0。
>
> 正则化会使权重变小。

L2正则化的实现主要分为两步：第一步就是在成本函数后面加个尾巴，第二步就是在计算偏导数时加个尾巴。在加入尾巴后，每次进行梯度下降，都会使得新的w变得更小，因此L2正则化也叫做权重衰减。可以这么理解，在加入L2正则化后，某些权重就变得很小，其相应的神经元就可以看作不起作用，从而使复杂的网络变得简单。但如果过度衰减，就可能反而造成欠拟合，因此就需要适度的选择正则化参数$\lambda$的值。

一般而言，如果$\lambda$值过高，则模型会非常简单，那么将面临欠拟合的风险，模型无法从训练数据中获得足够的信息来做出有用的预测；如果$\lambda$值过低，则模型会比较复杂，那么将面临过拟合的风险，模型将因获得过多训练数据个性方面的信息而无法泛化到新数据。

L2正则化的具体实现方法：

1. 在成本函数后面加个尾巴

   > ||W||指代的是权重向量(矩阵)W的范数。
   >
   > $\lambda$是一个超参数，称为正则化参数。
   >
   > m是样本数量。
   >
   > 成本函数尾巴 = 所有层的所有神经元的所有权重的平方和×$\frac{\lambda}{2m}$（常数乘常数）
   >
   > 权重W的梯度尾巴 = W×$\frac{\lambda}{m}$（矩阵乘常数）

   单神经元网络：<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514010715732.png" alt="image-20210514010715732" style="zoom:67%;" /> 

   多神经元网络：<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514010953700.png" alt="image-20210514010953700" style="zoom:67%;" /> 

2. 在计算权重的偏导数时加个尾巴：

   <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514011326535.png" alt="image-20210514011326535" style="zoom:67%;" /> 

## Dropout

Dropout（失活）和L2正则化一样，也是用于解决过拟合的一种正则化手段。

Dropout的原理就是随机删除神经元，即为每层设置一个概率数，用它来控制每一层有多少神经元应该被保留，0.8就意味着80％的被保留，20％的将被删除，1.0表示全部保留。每一层中哪些神经元应该被删除是随机的，所以神经网络每次训练都有不同的结构，而且不仅仅是每一次训练时不同，即使在同一次训练中不同样本所使用的网络结构也不同。

Dropout直接让神经网络变得简单，同时也让模型不会过分依赖于某一神经元，不会过分依赖某个神经元，就不会学习到个性，也就不会以偏概全，从而导致过拟合。

### 反向随机失活

反向随机失活是Dropout的一个实现。具体方法如下：

（假设某一层激活值矩阵为A，其梯度矩阵为dA）

1. 在前向传播时：

   ① 创建一个和A维度相同的随机矩阵D（称为掩码）：D=np.random.rand(A.shape[0], A.shape[1])；

   ② 令D=D<keep.prob。使得D中元素的值为0或1。其中，keep.prob为值为0-1的概率数，这里假设它的值为0.8，表示保留80%神经元，也就是D中只有80%的元素的值为1，其余为0；

   ③ 令A=A*D。这样就能让A中20%的元素变成0，也就相当于把对应的神经元删除了；

   ④ 令A=A/keep.prob。这句代码的原因：由于失活后预测值总体减少了20%，而测试时是使用完整神经网络的，因此测试时的预测值总体就不会缩小，这就导致训练和测试时的期望值不在一个水平上，使得测试不准确，无法用来衡量训练的好坏。所以，就需要对保留下来的预测值除以keep.prob，使其恢复原来的水平。

2. 在反向传播时：

   ① 令dA=dA*D。在前向传播时，我们通过掩码D与A进行运算而删除了一些神经元，而在反向传播时，我们也必须删除相同的神经元，这个可以通过让dA与相同的掩码D进行运算来实现；

   ② 令dA=dA/keep.prob。在前向传播时，我们将A除以了keep_prob，而在反向传播时，我们也必须将dA除以keep_prob。这是因为如果A被keep_prob进行了缩放，那么它的导数dA也应该被相应地缩放。

> 在前向传播和反向传播中都要实现dropout，即删除的神经元在这个回合的前向传播和反向传播中都不会起任何作用。
>
> dropout可以为每一层设置不同概率数。
>
> dropout有一个不好的影响。由于每次训练时的神经网络结构都不同了，所以成本不会随着训练的次数而递减了，这样的话就无法监视训练过程。为了避免这个缺陷，我们通常先会把dropout关掉，即把概率数设置为1，然后训练看一看成本是否呈下降的趋势，如果是，就说明代码没有问题，然后再开启dropout进行训练。
>
> 只能在训练模型时运行dropout，在使用模型时要把dropout关掉。

# 数据增强

为了解决数据不足，在实际开发中通常使用数据增强技术来生成伪数据，即通过调整原始样本来创建新样本，这样我们就可以获得大量的数据。这不仅增加了数据集的大小，还提供了单个样本的多个变体，有助于避免模型训练过度拟合。

1. 对于图片：水平翻转、旋转、扭曲、添加高斯噪声、擦除图像部分信息、颜色扰动（改变颜色分量或颜色通道顺序）。
2. 对于音频：调高或调低声音、将音频乘以随机因子以减少或增大音量、混入背景噪音。
3. 对于文本：回译（翻译为其他语言，再翻译回原语言）、同义词替换、随机插入、随即交换、随机删除。

# 特征规范化

特征规范化就是将数据输入的特征进行处理，使其满足某种规范，以便于神经网络进行计算，加快训练速度。下面介绍一种常见的规范化方法，它可以让样本分布映射为均值为0方差为1的标准正态分布，使得输入值落在激活函数比较敏感的区域（梯度大的区域），这种规范化属于线性变换，不会使得数据失活，也不会改变它们的数值排序。

假设数据集的每个样本有两个特征，下图每个点代表一个样本：

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514110735796.png" alt="image-20210514110735796" style="zoom: 50%;" /> 

1. 使数据集的平均值变成0。通过让每一个样本减去当前平均值来达到这个目的，下面的第一个公式是求当前平均值，第二个公式是减去平均值。经过变换后，样本分布图就变成右边所示，移动到了以0为中心的位置。

   <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514110542070.png" alt="image-20210514110542070" style="zoom:67%;" /> <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514110901141.png" alt="image-20210514110901141" style="zoom: 50%;" />

2. 使数据集的方差变成1，以减少数据集的离散程度（方差大离散程度就大）。通过下面两个公式实现。经过变换后，每个样本就都处在[-1,1]之间了。注意，这里的$x^{(i)}$是已经经过步骤1处理后的。

   <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514111136574.png" alt="image-20210514111136574" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514111148013.png" alt="image-20210514111148013" style="zoom: 50%;" /> 

> 总体过程总结：
>
> 1. 计算$ x^{(i)}$的平均值：$\mu=\frac{1}{m} \sum_{i=1}^{m} x^{(i)}$
> 2. 计算$ x^{(i)}$的均方差：$\sigma=\sqrt{\frac{1}{m} \sum_{i=1}^{m} x^{(i)^{2}}}$
> 3. 使$x$的分布变成以0为平均值，以1为方差的分布：$x=\frac{x-\mu}{\sigma}$
>
> 如果你对训练集的输入特征进行了规范化处理，那么你也必须对你的测试集以及实际应用中要预测的数据进行规范化处理。
>
> 在进行规范化处理时使用由训练集计算出来的u和σ，不要再根据测试集或实际待预测数据重新计算，只需要将数据直接减去u和σ。因为测试集和实际待预测数据的量都很小，由他们计算出来的u和σ是很片面的。

# 梯度消失和梯度爆炸

梯度消失：使偏导数（梯度）变得极端的小，导致权重变化极小，出现僵尸层，模型停止学习。梯度消失使得越靠前的网络层梯度越小，因此越靠前的网络层越容易变成僵尸层。出现原因：使用了不合适的激活函数，比如sigmoid。

梯度爆炸：使偏导数（梯度）变得极端的大，超出数值范围，导致无法继续学习。出现原因：网络层数多或权重的初始化数值太大。

一种解决方法：合理初始化权重。具体初始化公式如下所示，也就是在初始化权重时，采用符合正态分布的随机数，同时乘以一个依赖于上一层神经元数目的参数$np.sqrt(u/n^{[l-1]})$，以保证初始化后每个神经元的各个权重总和尽量靠近1。其中，n为上一层的神经元数，u是一个可调参数，一般来说，若激活函数是relu，就取u为2，若是tanh，就取u为1。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514112319122.png" alt="image-20210514112319122" style="zoom:67%;" /> 

> 保证初始化后每个神经元的各个权重总和尽量靠近1的原因：
>
> 大于1就会出现梯度爆炸，小于1就会出现梯度消失。可以考虑极端例子，比如a=g(wx+b)，其中g(x)=x,b=0，即a=wx，则最终的预测值可以表示为an=w1w2...wnx，从中能够看出来，如果w>1，则最后的预测值会越来越大，这就导致它的偏导数也会越大，反之越小，所以靠近1的话就会尽量延缓梯度消失或梯度爆炸的进程。

# 梯度检验

梯度检验用于检验反向传播中偏导数的计算是否出错，具体用数值逼近的方法来检验。

1. 假设我们要计算参数$\theta$对于成本函数$J$的梯度$\frac{\partial J}{\partial \theta}$；

2. 因为前向传播比较容易实现，所以我们确信我们的前向传播是正确的，即成本$J$是正确的。因此，我们就可以用计算成本$J$的前向传播代码来验证计算梯度$\frac{\partial J}{\partial \theta}$的反向传播代码是否正确；

3. 我们的目标就是验证$\frac{\partial J}{\partial \theta}$的准确性。为此，我们可以用另一种方式计算出$\frac{\partial J}{\partial \theta}$，如果它与反向传播计算得到的一样，那么反向传播就是正确的。这另一种计算方式就是偏导数的定义公式 $\frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon}$，通过这个公式，我们就可以用前向传播分别计算出$J(\theta + \varepsilon)$和$J(\theta - \varepsilon)$来求得$\frac{\partial J}{\partial \theta}$，进而验证反向传播；

4. 首先通过下面五小步来利用前向传播计算出梯度，这个梯度我们用$gradapprox$表示：

   $\theta^{+} = \theta + \varepsilon$（$\varepsilon$是一个小到非常靠近0的数，通常为$10^{-7}$）；

   $\theta^{-} = \theta - \varepsilon$；

   $J^{+} = J(\theta^{+})$；

   $J^{-} = J(\theta^{-})$；

   $gradapprox = \frac{J^{+} - J^{-}}{2 \varepsilon}$。

5. 接着利用反向传播也计算出一个梯度，这个梯度我们用$grad$表示；

6. 最后用下面的公式来计算$gradapprox$和$grad$这两个梯度相差多远；

   $difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2}$

7. 如果两个梯度的差距$difference$小于$10^{-7}$，那么说明这两个梯度很接近，也就是说，我们的反向传播是正确的；否则，就说明反向传播里面有问题；

8. 下面展示了多层多维参数网络的梯度检验部分的代码：

   - 将参数字典和反向传播得到的梯度字典分别转化为列向量$grad$；

     <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/dictionary_to_vector.png" alt="dictionary_to_vector" style="zoom: 33%;" /> 

   - 遍历每一个参数$θ_{i}$，并计算出$θ_{i}$对应的$gradapprox[i]$；

   - 计算出gradapprox和grad的差距，以此来判断反向传播是否正确。

```python
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    # 将参数字典转化为列向量
    parameters_values, _ = dictionary_to_vector(parameters)
    # 将反向传播计算的梯度字典也转化为列向量
    grad = gradients_to_vector(gradients)

    num_parameters = parameters_values.shape[0]
    # 创建存储结果的容器（列向量）
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # 遍历每一个参数，并计算出每个参数的gradapprox
    # 每个参数都对应了一个gradapprox
    # gradapprox是该参数对于成本函数的估计梯度
    for i in range(num_parameters):
        # 给参数加上epsilon
        thetaplus =  np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        # 利用前向传播计算出该参数加上epsilon后的成本
        J_plus[i], _ =  forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))

        # 给参数减去epsilon
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        # 利用前向传播计算出该参数减去epsilon后的成本
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))

        # 由上面计算得到的两个成本，计算出该参数的gradapprox
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # 求出gradapprox和grad的差距
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference > 2e-7:
        print("\033[93m" + "反向传播有问题! difference = " + str(difference) + "\033[0m")
    else:
        print("\033[92m" + "反向传播很完美! difference = " + str(difference) + "\033[0m")

    return difference
```

进行梯度检验时需要注意的一些事项：

1. 只在需要时才开启一下梯度检验。因为梯度检验计算量很大，一直开着浪费资源而且也没必要。
2. 检验出梯度偏差很大后，就进一步定位出是哪些参数相应的梯度有问题。打比方说，有时候只有某一层的db是有问题的，那么就需要重点检查那一层相关的结构、算法、代码。
3. 如果加了正则化尾巴（例如L2正则化），那么在梯度检验时也要带着尾巴。
4. 梯度检验和dropout不能同时使用。因为dropout每次都会随机删除神经元，梯度检验根本实施不起来。一般来说都是先把dropout关掉（即把概率数都设置为1），然后再使用梯度检验，检验完毕后再开启dropout。
5. 可以进行多次检验。例如在随机初始化参数后，可以进行一次梯度检验，然后让神经网络运行一阵把参数优化一些后，再进行一次梯度检验。

# HD5文件

HD5文件由group和dataset组成。HD5文件相当于一个文件夹，里面存放着group（目录）和dataset（文件），group相当于目录，因此group里面也可以继续存放group和dataset。dataset存储了具体的文件内容，包括数据集和描述该数据集的一些属性（metaData）。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514121832460.png" alt="image-20210514121832460" style="zoom:67%;" /> 

## 写入数据

```python
import h5py

"""
    create_dataset：新建 dataset
    create_group： 新建 group
"""

x = np.arange(100)

with h5py.File('test.h5','w') as f:
    f.create_dataset('test_numpy',data=x)
    subgroup = f.create_group('subgroup')
    subgroup.create_dataset('test_numpy',data=x)
    subsub = subgroup.create_group('subsub')
    subsub.create_dataset('test_numpy',data=x)
```

## 读取数据

```python
"""
    keys()：获取本文件夹下所有的文件及文件夹的名字
    f['key_name']：获取对应的对象    
"""
def read_data(filename):
    with h5py.File(filename,'r') as f:

        def print_name(name):
            print(name)
        f.visit(print_name)
        print('---------------------------------------')
        subgroup = f['subgroup']  
        print(subgroup.keys())
        print('---------------------------------------')
        dset = f['test_numpy']
        print(dset)
        print(dset.name)
        print(dset.shape)
        print(dset.dtype)
        print(dset[:])
        print('---------------------------------------')

read_data('test.h5')
```

# MiniBatch

MiniBatch就是将大训练集拆分成一个个小的训练集，然后依次用这些小的训练集来训练神经网络。如果使用MiniBatch使得成本在最小值附近徘徊，此时可以通过减小学习率来解决。

BatchSize（子训练大小，也是一个超参数）对网络的影响：

1. 没有BatchSize（也就是直接用整个训练集），梯度准确，只适用于小样本数据库。
2. BatchSize=1，梯度变来变去，成本在最小值附近徘徊，非常不准确，网络很难收敛。
3. Batch增大，梯度变准确。
4. BatchSize增大，梯度已经非常准确，再增加BatchSize也没有用。

下图分别是使用Batch梯度下降(蓝)、随机梯度下降(紫)和MiniBatch梯度下降(绿)的成本函数学习路径：

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514160818638.png" alt="image-20210514160818638" style="zoom:67%;" /> 

相关术语解释：

- MiniBatch梯度下降：用子训练集进行梯度下降；
- Batch梯度下降：用整个训练集进行梯度下降；
- 随机梯度下降：子训练集仅有一个样本的MiniBatch梯度下降；
- epoch：对整个训练集进行一次梯度下降就叫做一个epoch；
- iteration：进行一次梯度下降就叫做一个iteration。

MiniBatch梯度下降中最重要的就是对数据集进行重新洗牌和分割，当数据集被分割成一个个子数据集后，就可以通过循环遍历每一个子数据集来实现MiniBatch梯度下降。下面是对数据集进行洗牌和分割的代码实现：

```python
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)            
    m = X.shape[1]  # 获取样本数量
    mini_batches = []
        
    # 第一步：洗牌训练集
    # 对0~(m-1)的序列进行随机排序
    # 如果m是3，那么结果可能为[2, 0, 1]
    permutation = list(np.random.permutation(m))
    # X[行,列]
    # X[:,n]：表示把所有行中的第n个元素取出
    # X[n,:]：表示把第n行的所有元素取出
    # X[:,n:m]：表示把所有行中的从n到m-1的元素取出
    # X[n:m,:]：表示把从n到m-1行的所有元素取出
    # X[:,[3,2,1]]：表示把所有行的第3、2、1个元素按顺序取出，或者说是把第3、2、1列的元素取出
    shuffled_X = X[:, permutation] # 将X按permutation列表里的索引顺序进行重排列
    # 同时要保证标签和数据在洗牌前后还是一对
    # reshape((1,m))用于保证维度正确，因为有时候Python内部会自动改变维度
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # 第二步：分割洗牌后的训练集
    # 获取子训练集的个数（不包括后面不满mini_batch_size的那个子训练集）
    num_complete_minibatches = math.floor(m/mini_batch_size)
    # 循环取出满mini_batch_size的子训练集
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # 取出后面不满mini_batch_size的那个子训练集
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
```

# 动量梯度下降

## 指数加权平均

1. 加权平均

   把权重计算在内的平均方法，表示一组数据的平均水平。例如求n个数据的加权平均，其公式为$\sum_{i=1}^{n} 第i个数据的权重×第i个数据的值$。其中，权重指的是数据出现的频率。

2. 指数加权平均

   给定n个数据$d_{i}$和1个超参数k，求这n个数据$d_{i}$的指数加权平均。求法和一般的加权平均一样，都是权重与对应数值的乘积之和。不过这里的权重不再是数据的频率，而是一个与超参数k相关且随时间呈指数衰减的值，具体公式如下所示：

   $v_{n}=(1-k)d_{n}+(1-k)kd_{n-1}+(1-k)k^{2}d_{n-2}+...+(1-k)k^{n-1}d_{1}$

   求出来的平均值$v_{n}$在实际意义上并不是作为这n个数据的期望值，而是作为第n个数据在这n个数据中的趋势值。也就是说，如果用一条曲线来描述这n个数据在坐标轴上的趋势，$v_{n}$就是第n个数据在这条曲线上的值，$v_{n-1}$就是第n-1个数据在这条曲线上的值，且$v_{n-1}=(1-k)d_{n-1}+(1-k)kd_{n-2}+(1-k)k^{2}d_{n-3}+...+(1-k)k^{n-2}d_{1}$，对于$v_{n-2},v_{n-3},...,v_{1}$，以此类推。

   从上述的理解中，我们可以推出趋势值（指数加权平均）的递归表达式：$v_{n}=(1-k)d_{n}+kv_{n-1}$。

3. 超参数k的作用

   指数加权平均的表达式说明了第n个数据的趋势决定于自身的值和前n-1个数据的值。但实际上，第n个数据的趋势只会受其前$\frac{1}{1-k}-1$个数据的影响，因为更前面的数据由于指数衰减，权重已经非常小，它们的影响几乎可以忽略不计。换句话说，k的取值影响了趋势值能够受到其前面多少个数据的影响。k越大，则受到的影响越多，所得到的趋势曲线（由各个数据的趋势值连成的曲线）就越平滑；k越小，则受到的影响越少，所得到的趋势曲线变化剧烈，更贴近于原数据点的变化。

   > 黄色的线：k=0.5；红色的线：k=0.9；绿色的线：k=0.98

   <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514213357254.png" alt="image-20210514213357254" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514213411821.png" alt="image-20210514213411821" style="zoom:67%;" />

4. 偏差修正

   对于最开始的几个数据所求得的趋势值，与实际的数值会有较大的偏差，比如$v_{0}=0,k=0.9$，假设$d_{1}=40$，则$v_{1}=0.1×40+0.9×0=4$，与实际的40有非常大的偏差，因此我们需要对此进行修正，即把求得的趋势值$v_{i}$除以$1-k^{i}$。随着$i$的增加，这个除数会越来越接近于1，也就是说，只有靠前的数据需要被修正，靠后的数据的修正已经可以不考虑了。

## 在机器学习的应用

1. 具体方法

   在进行反向传播时，先求出偏导数，然后将其与之前训练得到的所有偏导数一起求指数加权平均，得出的趋势值代替偏导数进行梯度下降。这个方法就叫动量梯度下降。

2. 这样做的好处

   可以根据前几次的趋势来得出自己下降的方向，避免了走更多的弯路，相当于借鉴以往的经验。

   <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514214249739.png" alt="image-20210514214249739" style="zoom:67%;" /> 

3. 对动量的理解

   动量一般是指一个物体在它运动方向上保持运动的趋势。在动量梯度下降中，这个动量就是指w和b朝最小值这个方向保持运动的趋势。

4. 动量梯度下降比原版梯度下降的优势

   - 动量移动得更快（因为它积累的所有动量）；
   - 动量有机会逃脱局部极小值（因为动量可能推动它脱离局部极小值）；
   - 动量也将更好地通过高原区。
   
5. 代码实现

   ```python
   # 初始化指数加权平均值字典（容器）
   def initialize_velocity(parameters):
       # 获取神经网络的层数
       L = len(parameters) // 2
       v = {}
       
       # 循环每一层
       for l in range(L): 
           # 因为l是从0开始的，所以下面要在l后面加上1
           # zeros_like会返回一个与输入参数维度相同的数组，而且将这个数组全部设置为0
           # 指数加权平均值字典的维度应该是与梯度字典一样的，
           # 而梯度字典是与参数字典一样的，所以zeros_like的输入参数是参数字典
           v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l+1)])
           v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l+1)])
           
       return v
   
   # 使用动量梯度下降算法来更新参数
   def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
       L = len(parameters) // 2
       # 遍历每一层
       for l in range(L):
           # 从头递归计算出指数加权平均值
           v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
           v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
           
           # 用指数加权平均值来更新参数
           parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
           parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
           
       return parameters, v
   ```

# RMSProp

> RMSProp：Root Mean Square Prop：均方根传播，在对梯度进行指数加权平均的基础上，引入平方和平方根。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514232948150.png" alt="image-20210514232948150" style="zoom:67%;" /> 

假设梯度下降是在推一个球，让这个球不断靠近最小值处。这个球会沿上图中的锯齿路线不断地向最小值处移动。现在推这个球的有两个方向的力，一个横向力，一个纵向力，且参数w与横向的力有关，参数b与纵向的力有关（这里只是假设，实际有可能参数b与横向的力有关，也有可能w1和w2与横向的力有关，w3与纵向的力有关，因为参数w和b是多维的）。按照假设，我们可以通过RMSprop算法去改变w和b方向上的力的相对大小，从而影响球的走向。下面给出算法过程：

1. 首先计算出dw和db；

2. 然后算出它们平方后的指数平均值：$S_{dw}=kS_{dw}+(1-k)dw^{2}$，$S_{db}=kS_{db}+(1-k)db^{2}$；

   > 这里的平方是元素的平方，而不是矩阵相乘。

3. 更新参数w和b：$w=w-r(\frac{dw}{sqrt(S_{dw})})$，$b=b-r(\frac{db}{sqrt(S_{db})})$

   > sqrt表示开平方。

这个算法的原理就在于：如果球受到横向的里比较大，那么算法就会将w减小，但如果纵向的力比较大，那么算法就会将b减小。或者说，如果球受到横向的里比较小，那么算法就会将w增大，但如果纵向的力比较小，那么算法就会将b增大。也就是说，RMSprop算法会让球受到的各个方向的力不会过大也不会过小，而是尽量让它们的合力朝向最小值处。

此外，为了防止除数变为0，通常会让$sqrt(S_{dw})$变成$sqrt(S_{dw}+u)$，$sqrt(S_{db})$变成$sqrt(S_{db}+u)$。其中。u是一个很小的常数（通常是$10^{-8}$）。

# ADAM算法

> ADAM：Adaptive Moment Estimation

ADAM算法就是将动量梯度下降和RMSprop结合在一起。它的算法流程如下：

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210514234549520.png" alt="image-20210514234549520" style="zoom:67%;" /> 

> 第二步求出动量指数平均值（梯度的指数加权平均），第三步求出RMSprop指数平均值（梯度平方的指数加权平均）。因为动量梯度下降和RMSprop中都有一个超参数，所以将动量指数平均值中的超参数设为$k_{1}$，将RMSprop中的超参数用$k_{2}$表示。
>
> 第四步和第五步是对指数平均值进行修正。在使用Adam时会加上修正算法。公式中的t表示的是梯度下降的次数。
>
> 第六步是通过两个指数平均值来更新参数。

在Adam算法中有三个超参数r（学习率）和k1、k2。一般k1和k2分别设为09和0999，因此主要是调节学习率r，尝试不同的值，找到能让神经网络学得最快学得最准的学习率。

Adam算法可以自适应改变学习率。但实际上，学习率并没有被改变，改变的只是学习率右边的值，可是由于右边的值原本应该是dw的，它被指数平均值取代后，就相当于变相改变了学习率。比如说，原本的dw比较大，而它被较小的指数平均值取代后，就会使得w会变得更大，假设右边的值还是dw，那么就可以认为是学习率r变小了，从而让w变得比预期要大。

Adam算法的代码实现：

```python
# 初始化指数加权平均值变量（容器）
def initialize_adam(parameters) :    
    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])
    
    return v, s

# 使用adam来更新参数
# 参数epsilon用于防止除以0
# 参数t表示梯度下降的次数
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    # 存储修正后的值
    v_corrected = {}
    s_corrected = {}
    
    for l in range(L):
        # 算出v值
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        
        # 对v值进行修正
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        
        # 算出s值
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
    
        # 对s值进行修正
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
 
        # 更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)

    return parameters, v, s
```

# 对各种更新参数方法的解释

1. 随机梯度下降就像一个喝醉的人在摇摇晃晃地回家（全局最优点），走的弯路很多，因此回家也比较困难，花费的时间要很久。

2. 动量梯度下降就是在随机梯度下降的基础上，将喝醉的人所行走的平路变成斜坡，由于向下的惯性（动量），喝醉的人就会不自觉地朝下坡的方向前进更多，因此走的弯路就会比较少，从而更快到家。

3. AdaGrad也是在随机梯度的基础上进行改进，不过这次它不是将平路变成斜坡，而是每当喝醉的人要往某个方向行走一步时，把这个人的鞋子给替换掉（改变学习率）。如果这个人当前要走的方向不是朝着家的方向（即不是朝着全局最优点前进），则把他的鞋子换成不好走的鞋子（调低学习率），这样他在错误的方向上所走的距离就会变少；如果方向正确，也就是朝着家的方向（即朝着全局最优点前进），则把他的鞋子换成好鞋（调高学习率），这样他在正确的方向上所走的距离就会变多。不断重复换鞋，喝醉的人就能在弯路上不断少走几步，从而能够更快回家。

   <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210603150304684.png" alt="image-20210603150304684" style="zoom: 67%;" />

4. 将Momentum的惯性原则，加上Adagrad对错误方向的阻力，就能让喝醉的人更快回家，这就是RMSProp和ADAM的做法。不过在这个做法上，ADAM比RMSProp做得更加彻底，因此效果也更好。

# 学习率衰减

让学习率随着训练次数的增加而慢慢变小。下图是MiniBatch梯度下降，蓝色的线是固定学习率，而绿色的线采用学习率衰减的方式进行学习。可见，如果学习率固定在比较大的地方，模型的参数始终只能在最小值处徘徊。如果采用学习率衰减的方式，那么模型一开始学习的步进也比较大，等训练的次数多了，学习率降下来，学习的步进也减小了，即使参数在最小处徘徊，那也是在非常靠近最小值处的地方徘徊。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515001234643.png" alt="image-20210515001234643" style="zoom: 50%;" /> 

## 分数衰减

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515004727218.png" alt="image-20210515004727218" style="zoom:67%;" /> 

公式里面的decayRate是一个超参数，用它来控制学习率衰减的速度。epochNum是epoch所进行的数量。$r_{0}$是指初始的学习率，也是一个超参数。

从公式中可以看出，随着进行的epoch次数越来越多，学习率会越来越小。

## 分段常数衰减

分段常数衰减需要事先定义好的训练次数区间，在对应区间设置不同的学习率的常数值，一般情况刚开始的学习率要大一些，之后要越来越小，要根据样本量的大小设置区间的间隔大小，样本量越大，区间间隔要小一点。下图即为分段常数衰减的学习率变化图，横坐标代表训练次数，纵坐标代表学习率。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515005104259.png" alt="image-20210515005104259" style="zoom:67%;" /> 

## 指数衰减

以指数衰减方式进行学习率的更新，学习率的大小和训练次数指数相关，其更新规则为：

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515005305445.png" alt="image-20210515005305445" style="zoom:67%;" /> 

这种衰减方式简单直接，收敛速度快，如下图所示，绿色的为学习率随训练次数的指数衰减方式，红色的即为分段常数衰减，它在一定的训练区间内保持学习率不变。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515005417081.png" alt="image-20210515005417081" style="zoom:67%;" /> 

## 自然指数衰减

它与指数衰减方式相似，不同的在于它的衰减底数是自然数e，故而其收敛的速度更快，一般用于相对比较容易训练的网络，便于较快的收敛，其更新规则如下：

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515005545227.png" alt="image-20210515005545227" style="zoom:67%;" /> 

下图为为分段常数衰减、指数衰减、自然指数衰减三种方式的对比图，红色的即为分段常数衰减图（阶梯型曲线）。蓝色线为指数衰减图，绿色即为自然指数衰减图，明显可以看到自然指数衰减方式下的学习率衰减程度要大于一般指数衰减方式，有助于更快的收敛。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515005726206.png" alt="image-20210515005726206" style="zoom:67%;" /> 

## 多项式衰减

应用多项式衰减的方式进行更新学习率，这里会给定初始学习率和最低学习率取值，然后将会按照给定的衰减方式将学
习率从初始值衰减到最低值，其更新规则如下：

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515010039429.png" alt="image-20210515010039429" style="zoom: 50%;" /> 

需要注意的是，它有两个机制，一个是降到最低学习率后，直到训练结束都一直使用最低学习率进行更新，另一个是降到最低学习率后再次将学习率调高，使用$decay\_steps$的倍数，取第一个大于$global\_steps$的结果，须增加一个如下所示的式子。它是用来防止神经网络在训练的后期由于学习率过小而导致的网络一直在某个局部极小值附近震荡，这样可以通过在后期增大学习率跳出局部极小值。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515010159090.png" alt="image-20210515010159090" style="zoom:67%;" /> 

如下图所示，红色线代表学习率降低至最低后，一直保持学习率不变进行更新，绿色线代表学习率衰减到最低后，又会再次循环往复的升高降低。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515010308085.png" alt="image-20210515010308085" style="zoom:60%;" /> 

## 余弦衰减

余弦衰减就是采用余弦的相关方式进行学习率的衰减，衰减图和余弦函数相似。其更新机制如下式所示：

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515010741088.png" alt="image-20210515010741088" style="zoom:67%;" /> 

如下图所示，红色即为标准的余弦衰减曲线，学习率从初始值下降到最低学习率后保持不变。蓝色的线是线性余弦衰减方式曲线，它是学习率从初始学习率以线性的方式下降到最低学习率值。绿色的是噪声线性余弦衰减方式。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515010834652.png" alt="image-20210515010834652" style="zoom:60%;" /> 

# 局部最优问题

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515123154989.png" alt="image-20210515123154989" style="zoom:50%;" /> 

梯度下降可以看成球在山谷中滚动，并试图找到整体地势最低的地方，这个地方就叫做Global Minima，也叫全局最优。

局部最优就是除全局最优的那个坑以外的其余小坑，也被称为Loacl Minima。

鞍点就是指在部分维度上属于局部最优（梯度为0），但在其余维度上梯度不为0的地势，也就是Saddle Point。

## 局部最优已经不是问题

在低维神经网络（比如上图所示的二维网络）上，小球的确很容易被局部最优的坑所困而无法继续寻找全局最优。

但实际中的神经网络是很多维度的，要出现一个小坑，就必须满足某一点在所有维度上的梯度都为0，但这个概率是非常低的。此外，要让小球刚好滚动到小坑中又是一件非常低概率的事，两者叠加，可以认为小球被困在局部最优的情况几乎不会发生。

## 鞍点才是主要问题

相比于局部最优，鞍点出现的概率就比较大了，但鞍点并不会让小球困在原地，因为从上图可以看出，鞍点的横向维度是向上翘的，但是它的纵向维度是可以继续向下的。但鞍点上靠近底部的地方通常有一片特殊区域，这片区域叫做平稳段，由于这片区域的斜率比较小，小球在此处的滚动就会非常慢（即模型学习慢）。

如下图所示，小球可能会在横向花很长时间来到达鞍点的最小值处，然后再从这个点往纵向继续向下找全局最优处。鞍点的平稳段是无法避免的，因此只能利用前面所学的那些优化算法来加快学习，以更快度过那些平稳段。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210515125332787.png" alt="image-20210515125332787" style="zoom: 50%;" /> 

# 调参

## 超参数

超参数通常有以下这些：

1. 学习率r
2. 动量梯度下降中的超参数k
3. Adam中的超参数k、l、u
4. 神经网络层数L
5. 每层的神经元个数n
6. 学习率衰减控制超参数decayRate
7. 子训练集MiniBatch的大小

超参数间的重要性：

1. 学习率r

2. 动量梯度下降中的超参数k
   每层的神经元个数n
   子训练集minibatch的大小

3. 神经网络层数L

   学习率衰减控制超参数decayRate

4. 剩余的超参数

## 随机搜索法

随机搜索法（也叫随机均匀采样）是一种调参手段。比如要调的超参数有两个，则它们的搜索范围就是一个矩形，然后在这个矩形内随机选取一些点作为超参数的值。如果发现某个区域内的点的值都不错的话，那么就可以把搜索范围缩小到那个区域，然后对那个区域进行更加精密的搜索。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210521155943806.png" alt="image-20210521155943806" style="zoom:46%;" /><img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210521155954239.png" alt="image-20210521155954239" style="zoom:50%;" />

> 采样是指从一定范围的总体中抽取个体的过程。
>
> 选择合适的采样标尺（采样范围）：
>
> 1. 采样标尺是指将某一个整体的采样范围划分为好几段，然后在这几段小的范围内分别进行采样，而这个划分采样范围的标准就叫采样标尺；
>
> 2. 线性标尺：被划分的几段采样范围的长度呈线性关系；
>
> 3. 指数标尺：被划分的几段采样范围的长度呈指数关系。指数标尺通常针对那些对变化比较敏感的超参数；
>
> 4. 对于网络层数、神经元个数等超参数，通常使用线性标尺；
>
>    <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210521162146010.png" alt="image-20210521162146010" style="zoom: 50%;" /> 
>
> 5. 对于学习率、动量梯度下降中的参数(1-k)等超参数，通常使用指数标尺。
>
>    <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210521162230547.png" alt="image-20210521162230547" style="zoom:50%;" /> 

## 调参经验

1. 学习率设定

   一般学习率从0.1或0.01开始尝试。学习率设置太大会导致训练十分不稳定，甚至出现NaN；设置太小会导致损失下降太慢。学习率一般要随着训练进行衰减。如果选用了Adam作为优化器，则它们自带了可自适应的学习率。

2. MiniBatch

   一般为2的次方，比如32、64、128、256等。较大的MiniBatch使得计算速度较快，但内存可能超出，且太大的MiniBatch反而使准确率下降；较小的MiniBatch可能因为收敛的抖动比较厉害，反而不容易卡在局部最优点，但会降低训练速度。

3. Epoch次数

   如果随着Epoch的次数增加，准确度在一定时间内变化很小，就可以停止Epoch，防止过拟合。

4. 隐含层神经元数

   解决问题的模型越复杂则用越多Hidden Units，但是要适度，因为太大的模型会导致过拟合。通常增加Hidden Units的数量直到Validation Error变差。

5. 激活函数的选择

   常用的激活函数有relu、leaky-relu、sigmoid、tanh等。对于输出层，多分类任务选用softmax输出，二分类任务选用sigmoid输出，回归任务选用线性输出。而对于中间隐层，则优先选择relu激活函数，因为relu函数可以有效解决sigmoid和tanh出现的梯度弥散问题，它会比其他激活函数以更快的速度收敛。

6. 数据集大小：越大越好。

7. dropout：通常为0.5。

8. 梯度裁剪

   用于解决梯度爆炸：如果梯度$g$变得非常大，那么就调节它使其保持较小的状态，即如果$||g||_{2}≥c$，则令$g = c *g /||g||_{2}$。其中，c是一个阈值超参数。$g /||g||_{2}$必然是个单位矢量，因此在进行调节后，新的梯度$g$的范数必然等于c，也就是说，梯度裁剪确保了梯度矢量的最大范数。

> 除以上外，还有一些调参技巧：
>
> 1. 保证正负样本的比例平衡。
> 2. 学会参考：各个机器学习领域间存在相关性，某个领域的调参技巧可能同样适用于其他领域。因此要多阅读其他领域的论文来获取好想法。
> 3. 超参数会过时。
> 4. 画图：准确率图、loss图。
> 5. 从小规模数据、复杂模型开始，直接奔着过拟合，若这样都没有效果，则可以反思是不是模型存在问题？
> 6. 观察loss胜于观察准确率。

## 调参模式

### 熊猫宝宝式

熊猫宝宝式就是指在同一时刻只训练一个或几个模型来进行实时调参。通常用在硬件资源比较缺失，而无法同时训练很多个模型的情况下。

使用熊猫宝宝式，需要我们不断地观察模型状况，不断地尝试调整不同的超参数值。就像照顾熊猫宝宝一样，发现它饿了就给东西吃，渴了就给它水喝。打个比方：

最初我们会随机初始化一些超参数值，然后观察模型的表现状况（比如损失曲线或验证集的错误率曲线）。如果发现模型第一天的学习状况不错，那么就可以尝试将学习率再调大一点点，看看模型的表现会是如何，可能会学得更快了，也可能会变得糟糕起来了。如果第二天你发现模型的学习状况依然不错，这时你可以尝试加入动量梯度下降算法，看看模型的表现又会是如何。就这样，每天你都要根据模型的学习状况来尝试不同的超参数值。如果某一天你发现模型的表现变差了，那么你可以将超参数返回到前一天的值。

### 鱼卵宝宝式

如果硬件资源充足，那么可以考虑鱼卵宝宝模式，即同时训练海量的模型。每一个模型都被设置了不同的超参数值，然后就不需要管它们了，让这些模型在服务器上一直训练下去。这些模型中有些可能会表现得很好，有些可能会表现得很差，到时候我们就选择那些使得模型表现好的超参数值就行了。就像鱼类一样，一次性会产很多的卵，之后就让这些小鱼自身自灭，有些小鱼可能活不了，但是总有一些会长成大鱼。如果有足够的计算资源的话，使用鱼卵模式会更加轻松高效。

>一些调参工具：
>
>- [Ray.tune]([ray/python/ray/tune at master · ray-project/ray (github.com)](https://github.com/ray-project/ray/tree/master/python/ray/tune))
>- [NNI]([microsoft/nni: An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning. (github.com)](https://github.com/microsoft/nni))
>- [Chocolate]([AIworx-Labs/chocolate: A fully decentralized hyperparameter optimization framework (github.com)](https://github.com/AIworx-Labs/chocolate))
>- [HORD]([ilija139/HORD: Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates (github.com)](https://github.com/ilija139/HORD))

# Batch Normalization

> 论文出处：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

Batch Normalization是指对隐藏层的输出z（或经过激活后的a）进行规范化处理。BN经常使用在MiniBatch上，这也是其名称的由来。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/BN.png" alt="BN" style="zoom: 80%;" />

## BN的好处

1. 每层对z进行规范化可以把激活输入值控制在非线性函数（激活函数）对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，避免梯度消失的问题产生，加速收敛。
2. 使隐藏层z值的变化幅度更小，更加稳定，从而让输入分布固定下来，减小ICS带来的影响（限制了前层参数更新对数值分布的影响程度，减少了各层W和b之间的耦合性，让各层更加独立），让模型的鲁棒性更强。
3. 能够起到微弱的正则化效果。因为在每个MiniBatch而非整个数据集上计算均值和方差，只由这一小部分数据估计得出的均值和方差会有一些噪声，因此最终计算出的z也有一定噪声。类似于dropout，这种噪声会使得神经元不再特别依赖于任何一个输入。
4. 因为BN只有微弱的正则化效果，因此可以和dropout一起使用，以获得更强大的正则化效果。
5. 通过应用更大的MiniBatch 大小，可以减少噪声，从而减少这种正则化效果。
6. BN可以使参数搜索问题变得更容易，使神经网络对超参数的选择更加稳定，而超参数的采样范围也会变得更大。

>Internal Covariate Shift（ICS）：数据集的输入特征总是不变的，所以对输入层来说，它接收到的输入分布是不变的。但隐藏层接收到的输入数据每次都会不同（因为w和b每次都不同，使得z每次也不同），这种输入分布总是变来变去的状态，就被称为ICS。
>
>神经元很难从变来变去的输入分布中学到东西，因此将输入分布固定下来可以加速神经元的学习。

## BN的具体方法

1. 计算$z^{(i)}$的平均值：

   $μ=\frac{1}{m} \sum_{i=1}^{m} z^{(i)}$

2. 计算$z^{(i)}$的方差：

   $\sigma=\frac{1}{m} \sum_{i=1}^{m}\left(z^{(i)}-μ\right)^{2}$

3. 使$z^{(i)}$的分布变成以0为平均值，以1为方差的分布：

   $z^{(i)}_{norm}=\frac{z^{(i)}-μ}{\sqrt{\sigma+\varepsilon}}$

4. 使z的分布区域可以被任意挪动：

   $\tilde{z}^{(i)} =\gamma z^{(i)}_{norm}+\beta$

> $\gamma$和$\beta$是与w和b一样的模型学习参数，也会随着神经网络的学习而不断被优化。
>
> $\gamma$用于控制方差，$\beta$用于控制平均值。比如设$\gamma$为$\sqrt{\sigma+\varepsilon}$，$\beta$为$u$，就把第一、二步对z的平均值和方差的修改给抵消了。也就是说，通过设置$\gamma$和$\beta$，就可以控制z的平均值和方差。
>
> 设置$\gamma$和$\beta$的原因是，如果各隐藏层的输入均值在靠近0的区域，即处于激活函数的线性区域，不利于训练非线性神经网络，从而得到效果较差的模型。因此，需要用$\gamma$和$\beta$对标准化后的结果做进一步处理。
>
> 使用BN时，因为标准化处理中包含减去均值的一步，因此阈值b实际上没有起到作用，其数值效果交由$\beta$来实现。因此，在BN中，可以省略 b 或者暂时设置为 0。
>
> 在使用梯度下降算法时，分别对w、$\gamma$和$\beta$进行迭代更新。

## 关于BN的注意事项

1. 与特征规范化一样，BN也是对输入进行规范化处理，因为当前隐藏层的输出z其实就是下一层的输入。总的来说就是，在神经网络中，我们会为数据集进行规范化处理，然后对第一层的z进行规范化处理，接着对第二层的z进行规范化处理，直到最后一层。

2. BN既可以用于对z进行处理，也可以用于对a进行处理。

3. BN的启发来自于白化操作。白化就是指将输入数据的分布变化到以0为均值，以1为方差的正态分布，以加速收敛。

4. 不要将BN作为正则化的手段，而是当作加速学习的方式。

5. 如何得到使用模型时进行BN规范化处理所需要的u和σ？

   根据最后一次训练时得到的所有子训练集mini-batch的u'和σ'，计算得出它们的指数加权平均值u和σ，然后用u和σ进行使用模型时的规范化处理。

   比如，如果$u_{1}[1]$是第1个子训练集第1层神经网络相关的u，$u_{2}[1]$是第2个子训练集第1层神经网络相关的u，......，那么在使用模型时，第1层神经网络相关的u就是$(u_{2}[1],u_{2}[1],...u_{n}[1])$的指数加权平均值。

# Softmax

Softmax是一种激活函数，用于实现神经网络的多分类问题。

> 二分类问题：神经网络输出层只有一个神经元，其输出y'表示预测样本是正类的概率，若y'> 0.5则判断为正类，反之则判断为负类。
>
> 多分类问题：假设需要判断的类别有C个，则神经网络的输出层有C个神经元，每个神经元代表一个类别，由这C个神经元分别给出输入样本属于该神经元对应类别的概率，且这C个概率的总和等于1。通常取概率最大的那个类别作为输入样本的预测类别。

Softmax实现多分类问题的具体方法：

1. 用形状为(C,1)的列向量z表示整个输出层的输出，其每个元素代表一个神经元$i$的输出$z_{i}$；
2. 令临时变量$t=e^{z}$，$t$也是一个形状为(C,1)的列向量；
3. 计算出激活值$a=\frac{t}{np.sum(t)}$，$a$也是一个形状为(C,1)的列向量，其每个元素代表一个神经元$i$的输出的激活值$a_{i}$。同时，$a_{i}$也表示了输入样本属于该元素对应神经元所代表类别的概率。

这上面的整个过程可以用一个公式总结：$a_{i}=\frac{e^{z_{i}}}{\sum_{i=1}^{C} e^{z_{i}}}$。

>  一个直观的计算例子如下：
>
> <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210522010342429.png" alt="image-20210522010342429" style="zoom: 67%;" />

多分类神经网络的损失函数和成本函数：

>在多分类问题中，样本的标签定义为形状为(C,1)的列向量y，其中只有一个元素是1（表示该样本属于该类），其余为0。假设神经网络的输出为y'，是一个形状为(C,1)的概率向量，只有当概率最大的那个元素所代表的类别与标签y中元素为1所代表的类别一致时，才说明预测准确。
>
>多分类神经网络的梯度下降和二分类网络的步骤完全一致。

1. 损失函数定义为：$L(y', y)=-\sum_{j=1}^{C} y_{j} \log y'_{j}$

2. 由于标签y中只有一个元素是1，假设为$y_{i}$，则损失函数可以简化为：

   $L(y', y)=-y_{i} \log y'_{i}=-\log y'_{i}$

   （梯度下降的目的就是让损失函数最小，也就是$-\log y'_{i}$最小，即$y'_{i}$最大。也就是说，$y'_{i}$越大，预测就越精准。因此通过这个损失函数，梯度下降就会一步步让$y'_{i}$越来越大，从而使预测值越来越接近真实标签值）

3. m个样本的成本函数定义为：$J=\frac{1}{m} \sum_{i=1}^{m} L(y', y)$

> 选择Softmax分类器还是二元分类器？
>
> - 取决于类别之间是否互斥。
>
> - 举个例子，若有个任务是将图像分到三个不同类别中：
>
>   (1) 假设这三个类别分别是：室内场景、户外城区场景、户外荒野场景；
>
>   (2) 假设这三个类别分别是室内场景、黑白图片、人物图片；
>
>   在(1)中，这三个类别是互斥的（一个图像只能是这三个类别之一），因此更适于选择Softmax回归分类器。而在(2)中，这三个类别是互相包含的（一个图像可能属于多个类别），因此为其分别建立三个logistic二元回归分类器更加合适，这样就可以分别判断它属于哪些类别。

