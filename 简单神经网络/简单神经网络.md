# 机器学习

机器学习的目的就是找到一组使得成本函数最小的参数，其过程一般分为前向传播和反向传播。

> 机器学习的目的就是找到一个good function。
>
> 机器学习就是让机器不断的尝试模拟已知的数据，它能知道自己拟合的数据离真实的数据差距有多远，然后不断地改进自己拟合的参数，提高拟合的相似度。

1. 前向传播

   ① 计算预测值：逻辑回归（(特征×权值)的总和+阈值） -> 激活函数（映射到0-1）

   ② 评价是否准确：损失函数

2. 反向传播

   ① 计算损失函数关于权值和阈值的偏导数d

   ② 梯度下降，即利用偏导数确定新的权值和阈值：x'=x-rd（r为学习率）

   ③ 利用新的权值和阈值重新进行前向传播

在反向传播中如何求偏导数？

- 先写出前向传播的公式，然后根据链式法则，写出每一步的偏导数，然后进行链式相乘即可。
- 推导偏导数的公式时尽量朝最小化的情况（公式不出现向量）进行推导，推导完成后只需在最终公式上扩展成向量形式即可。
- 如果某一层的某个参数a对后一层的多个神经元都有贡献，则后一层对该参数a的总偏导就是这多个神经元分别对a的偏导的总和（雅可比矩阵）。
- 画出传播图。
- 在实际写代码时，可以==只写单样本的情况==，多样本的向量化一般在写完单样本的情况后就自动完成了。此时要注意的是前后数据的形状变化，最有效的运算目标就是==朝着目标数据的形状出发==，只要你所写代码运算后的形状与目标数据的形状一致，一般就是成功的。通常情况下，==每一层输出的形状都要保持一致==，要么是(sample_count, output_count)，要么是(output_count, sample_count)，具体是哪个要看自己的选择（一般是前者）。

张量是什么？

- 零维张量：标量
- 一维张量：向量
- 二维张量：矩阵

方向导数和梯度

（假设多元函数f(x)，x是向量，其中每个元素代表函数f的一个自变量）

- 梯度：包含所有自变量的偏导数的向量，记作$▽_xf(x)$。
- 方向导数：若有一个向量a，则函数f在这个向量的方向上的变化率称为函数f关于向量a的方向导数$▽_af(x)$，公式为$▽_af(x)=E(a)^T·▽_xf(x)$（E(x)表示将a变成单位向量）。
- 方向导数为正时，代表增加或者上升，正得越多，上升越快；方向导数为负时，代表减小或者下降，负得越多，下降越快。因此方向导数越小，即负得越多，则下降越快，为了求下降最快的方向，我们就令方向导数最小，即$min▽_af(x)$，这时可以发现在梯度向量的反方向上是下降最快的。
- 梯度的方向导数是在所有方向导数中变化最快的，或者说，往梯度的反方向上前进，可以最快到达函数f的最低点，这就叫最速下降法。最速下降在梯度的每一个元素（偏导数）为0时收敛。
- 有时候，可以通过直接求解▽f=0直接跳到最低点。

最简单的神经网络就是由一个个感知机（也就是神经元）组合而成的。

# 单神经元网络

## 依赖py库

numpy：提供向量运算功能

h5py：读取h5类型的数据集

matplotlib：数据可视化

skimage：将任意图片转化成能够输入模型的格式

## 基本策略

在训练模型时，一次传播计算出所有样本的损失值，然后取其平均值进行梯度下降。重复这一过程足够多次。

## 数据清洗

- 任意图片由matplotlib读取后均为三维矩阵(x,y,rgb)形式的数据。

  > x和y为像素点的横纵坐标，rgb为某个像素点的RGB三通道的秩3数组。

- 由skimage转换为(64,64,3)形式的三维数据。

- 由numpy转换为(64×64×3,1)形式的列向量，每个元素（RGB通道）代表一个样本的特征。

## 模型训练

目标模型：预测给定图片是否存在猫。

> 公式是针对某一样本的计算过程，py代码是针对所有样本的向量化计算过程。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/TIM%E5%9B%BE%E7%89%8720180630153721.png" alt="TIM图片20180630153721" style="zoom:67%;" /> 

### 初始化参数

权重和阈值均初始化为0。

### 前向传播

1. 求预测值

   公式：$$z = w_{1}x_{1}+w_{2}x_{2}+...+w_{n}x_{n} + b$$

   py代码：$Z = np.dot(W, X) + b$ 

   > W：行权重向量，每个元素代表一个特征的权重
   >
   > X：列特征向量矩阵，每个列向量代表一个样本，列向量的每个元素代表样本的一个特征
   >
   > b：阈值
   >
   > Z：行预测值向量，每个元素代表一个样本的预测值

2. 激活

   公式：$a=\frac{1}{1 + e^{-z}}$ 

   py代码：$A = 1 / (1 + np.exp(-Z))$ 

   > Z：行激活向量，每个元素代表一个样本预测值被激活后的数值

3. 求平均损失值

   公式：$l=-yln(a)-(1-y)ln(1-a)$ 

   py代码：$L_{avg} = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / n$

### 反向传播

1. 求偏导数（可见实际计算过程与平均损失值无关）

   > 推导过程：$dl/da=-\frac{y}{a}+\frac{1-y}{1-a}$，$da/dz=a(1-a)$，$dz/dw_{i}=x_{i}$，$dz/db=1$ 

   公式：$dl/dz=a-y$，$dl/dw_{i}=(a-y)x_{i}$，$dl/db=a-y$ 

   py代码：$dZ = A - Y$，$dW = np.dot(X,dZ.T) / n$，$  db = np.sum(dZ) / n$ 

   > dZ：行向量，每个元素代表一个样本的dl/dz值
   >
   > dW：列向量，每个元素代表一个权重的平均偏导数
   >
   > db：平均阈值

2. 梯度下降

   公式：$w_{i}=w_{i}-k×dw_{i}$，$b=b-k×db$ 

   py代码：$W=W-k×dW.T$，$b=b-kdb$

## 快速训练

利用py库sklearn可以快速进行逻辑回归单神经元网络的模型训练。

```python
# 取得一个逻辑回归单神经元网络的模型
LR_model = sklearn.linear_model.LogisticRegressionCV();
# 输入训练集进行训练
LR_model.fit(X, Y); # X为行向量矩阵(n_samples, n_features)，Y为[n_samples]（可以使用ravel()将矩阵展成一维数组）。训练好的w和b参数保存在对象LR_model中。
LR_predictions = LR_model.predict(Z) # Z为(n_samples, n_features)。返回值为元素是预测值的数组[n_samples]
# 详情参考：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
```

# 两层神经网络

## 基本策略

多层神经网络的训练由多个单神经元的传播叠加而成，首层神经元的输入由训练集提供，前一层神经元的输出作为后一层神经元的输入，最后一层一般只有一个神经元，其输出作为最终的预测值。

相关知识：多层神经网络前向传播、反向传播、激活函数的意义、随机初始化参数的意义。

<img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/%E5%A4%9A%E5%B1%82%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD.png" alt="多层神经网络前向传播" style="zoom:5%;" />  <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/%E5%A4%9A%E5%B1%82%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.png" alt="多层神经网络反向传播" style="zoom:5%;" /> <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E7%9A%84%E6%84%8F%E4%B9%89.png" alt="激活函数的意义" style="zoom:5%;" /> <img src="https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/%E9%9A%8F%E6%9C%BA%E5%88%9D%E5%A7%8B%E5%8C%96%E5%8F%82%E6%95%B0%E7%9A%84%E6%84%8F%E4%B9%89.png" alt="随机初始化参数的意义" style="zoom:5%;" /> 

## 模型训练

目标模型：预测直接坐标轴上某一个坐标对应的颜色。

> 不再出现已推导的公式。

![figure](https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/figure.png) 

### 初始化参数

保证各神经元之间的权重不同，而阈值均为0即可。

### 前向传播

1. 第一层（包含4个神经元，每个样本的2个特征同时作为这4个神经元的输入，它们的输出作为下一层神经元的输入）

   ① 求输出：$Z1 = np.dot(W1, X)+B1$ 

   ② 激活：$A1 = np.tanh(Z1)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}|_{x=Z1}$ 

   > W1：第一层的权重。行向量矩阵，每个行向量 => 一个神经元的权重，行向量的每个元素 => 该神经元一个输入的权重。
   >
   > B1：第一层的阈值。列向量，每个元素 => 一个神经元的阈值。
   >
   > X：训练集，每个样本的特征作为第一层的输入。列向量矩阵，每个列向量 => 一个样本，列向量的每个元素 => 样本的一个特征。
   >
   > Z1：第一层的输出。行向量矩阵，每个行向量 => 一个神经元在所有样本下的输出。
   >
   > A1：第一层激活后的输出。

2. 第二层（仅有1个神经元，上一层四个神经元的输出作为该神经元的输入，它的输出作为最终的预测值）

   ① 求输出：$Z2 = np.dot(W2, A1)+b2 $

   ② 激活：$A2 = sigmoid(Z2)=1/(1+np.exp(-Z2)) $

   > W2：第二层的权重。行向量，每个元素 => 第二层神经元输入的一个权重。
   >
   > b2：第二层的阈值。
   >
   > A1：第一层激活后的输出。
   >
   > Z2：第二层的输出。行向量，每个元素 => 一个样本的预测值。
   >
   > A2：经过激活后的最终预测值。

3. 求平均损失值

   $L_{avg} = -np.sum(np.multiply(Y, np.log(A2))+np.multiply(1-Y, np.log(1-A2)))/n$

### 反向传播

以损失函数为起点，从最后一层向前逐层地计算每个神经元的权重和阈值的偏导数，然后进行梯度下降。

1. 求第二层神经元的权重和阈值的偏导数

   $dZ2 = A2 - Y$

   $dW2 = (1 / n) * np.dot(dZ2, A1.T)$

   $db2 = (1 / n) * np.sum(dZ2)$

2. 求第一层神经元的权重和阈值的偏导数

   > 公式推导（仅考虑单样本情况，在实际应用时通过向量化拓展到多样本情况）：
   >
   > 1. 第二层
   >
   >    $dl/dz_{2}=a_{2}-y$
   >
   >    $dl/dw_{2i}=(a_{2}-y)a_{1i}$
   >
   >    $dl/db_{2}=a_{2}-y$ 
   >
   > 2. 第一层
   >
   >    $dz_{2}/da_{1i}=w_{2i}$
   >
   >    $dl/dz_{2}=a_{2}-y$
   >
   >    $dl/da_{1i}=w_{2i}(a_{2}-y)$
   >
   >    $da_{1i}/dz_{1i}=1-a_{1i}^{2}$
   >
   >    => $dl/dz_{1i}=w_{2i}(a_{2}-y)(1-a_{1i}^{2})$
   >
   >    $dz_{1i}/dw_{1i}=x_{1i}$ 
   >
   >    $dz_{1i}/db_{1i}=1$ 
   >
   >    => $dl/dw_{1i}=\frac{dl}{dz_{1i}} x_{1i}$ 
   >    
   >    => $dl/db_{1i}=\frac{dl}{dz_{1i}}$ 

   $dA1=np.dot(W2.T, dZ2)$

   $dZ1 = np.multiply(dA1, 1 - np.power(A1, 2))$ 

   $dW1 = (1 / n) * np.dot(dZ1, X.T)$ 

   $dB1 = (1 / n) * np.sum(dZ1, axis=1, keepdims=True)$ 

3. 梯度下降（k为学习率）

   $W1 = W1 - k * dW1$

   $B1 = B1 - k * dB1$

   $W2 = W2 - k * dW2$

   $$b2 = b2 - k * db2$$

# 深度神经网络

![image-20210527165045358](https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/image-20210527165045358.png)

## 基本策略

在两层神经网络的基础上继续增加网络层数，其计算方式和两层神经网络完全一样，具体如下所示：

1. 每层神经元的计算流程：计算回归值 => 激活回归值；
2. 末层神经元的计算流程：计算回归值 => 激活回归值 => 计算成本或损失；
3. 第一层的输入为样本特征值，其他层的输入为前一层的回归激活值。

## 模型训练

目标模型：预测图片上是否有猫。

### 初始化参数

权重随机取值，阈值取零。

### 前向传播

① 计算预测值：

```python
# 前layerCount-1层
A = X
for i in range(1, layerCount):
	A_prev = A
	Z = np.dot(W[i], A_prev) + b[i]
	A = relu(Z)

# 最后一层
Z_final = np.dot(W_final, A) + b_final
A_fianl = sigmiod(Z_final)
```

② 计算成本：

$cost = (-1 / n) * np.sum(np.multiply(Y, np.log(A_{final})) +np.multiply(1 - Y, np.log(1 - A_{final})))$

### 反向传播

① 求偏导数

```python
# 最后一层
dA_final = np.divide(1-Y, 1-A_final) - np.divide(Y, A_final)
dZ_final = dA_final * A_final * (1 - A_final)
# 为什么是A_prev?
# 因为z=wx+b，对w求导，w没了，留下x，所以是A_prev
dW_fianl = np.dot(dZ_final, A_prev.T) / n_final
db_final = np.sum(dZ_final, axis=1, keepdims=True) / n_final

# 其他层
dZ_back = dZ_final
W_back = W_final
for i in reversed(range(1, layer_count)):
    # dL/dZ_back = dZ_back
    # dZ_back/dA = W_back
    # 为什么是W_back?
    # 因为求当前层A的偏导用的是后一层的回归函数，所以是W_back
    dA = np.dot(W_back.T, dZ_back) # Note: 当前层a的总梯度 = 后一层多个神经元对a的梯度的总和
    dZ = np.array(dA, copy=True)
    dZ[Z[i] <= 0] = 0
    dW = np.dot(dZ, A_prev.T) / n[i]
    db = np.sum(dZ, axis=1, keepdims=True) / n[i]

    dZ_back = dZ
    W_back = W[i]
```

② 更新参数

```python
for i in range(1, layer_count+1):
    params['W'+str(i)] -= learning_rate*grads['dW'+str(i)]
    params['b'+str(i)] -= learning_rate*grads['db'+str(i)]
```

