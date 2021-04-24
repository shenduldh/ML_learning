# 识猫

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

![TIM图片20180630153721](https://cdn.jsdelivr.net/gh/shenduldh/image@main/img/TIM%E5%9B%BE%E7%89%8720180630153721.png)

### 初始化参数

权重和阈值均初始化为0。

### 前向传播

1. 求预测值

   公式：$z = w_{1}x_{1}+w_{2}x_{2}+...+w_{n}x_{n} + b$ 

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

   公式：$l=-yln(a)-(1-y)ln(1-a)$$  

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

   $dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))$ 

   $dW1 = (1 / n) * np.dot(dZ1, X.T)$ 

   $dB1 = (1 / n) * np.sum(dZ1, axis=1, keepdims=True)$ 

3. 梯度下降（k为学习率）

   $W1 = W1 - k * dW1$

   $B1 = B1 - k * dB1$

   $W2 = W2 - k * dW2$

   $$b2 = b2 - k * db2$$
