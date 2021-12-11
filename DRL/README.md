# 什么是强化学习

<img src="https://pic1.zhimg.com/80/v2-bd73e582bbcd395298d219ceac3f4a20_720w.jpg" alt="img" style="zoom: 50%;" />

强化学习就是让一个智能体（agent）在不同的环境状态（state）下，学会选择那个使得奖赏（reward）最大的动作（action）。

agent可以看作是一个机器人，这个机器人在某一时刻，通过观测environment（环境）得到自己所处的state（状态），然后根据policy（策略）进行一些运算（相当于思考和决策）之后，做出一个action（动作）。这个action会作用在environment上，使得agent在environment中可以从当前的state转移到一个新的state，并且在转移时获得一个即时的reward（奖赏）。由于转移到不同的state所获得的reward不一定一样，因此agent可以通过选择不同action来获得不同的reward。不断循环这一过程，agent就可以累积很多reward，而agent的目标就是在到达终点时获得最多的reward累积。

总的来说，强化学习就是一个agent与environment不断交互地过程，而交互的目的就是为了获得尽可能多的reward。此外，强化学习是一个反复迭代的过程，每一次迭代要解决两个问题：给定一个策略求值函数，和根据值函数来更新策略。

强化学习的一些概念解释：

1. Policy：指agent选择动作的策略，即agent根据这个策略来选择动作。在强化学习中，agent更具体的目标就是通过学习来优化policy，最终使得agent用这个policy来选择动作时可以使得agent获得尽可能多的reward。

2. Reward：用于告诉agent所选择的动作是好是坏，奖励最大化也就是agent的目的。

3. State：包括两方面的state，即环境状态和观测状态。

   environment state（$S^{e}_{t}$）：环境的客观状态，但不一定能被agent全面地观测到，因此与agent观测到的状态有所区别。agent执行一个动作之后，环境的状态也会有所变化，然后根据环境的状态产生reward反馈给agent。

   agent state（$S^{a}_{t}$）：环境的主观状态，也就是agent观测到的状态。这个状态只需要有利于agent的目标即可，比如agent只需要一个摄像头看清前面的路就可以向前行走了，并不需要知道声音。也就是说，agent观测一部分状态就可以做它想要完成的目标。如果agent能够完全观测到环境的状态，则$S^{a}_{t}=S^{e}_{t}$。

4. Value Function：reward表示了agent当前所选择动作的好坏，而value function则是从长期的角度来看agent选择动作的好坏。

5. Exploration（探索）：指做你以前从来没有做过的事情，以期望获得更高的回报。

6. Exploitation（利用）：指做你当前知道的能产生最大回报的事情。

# 基于内在动机的强化学习

在许多真实的场景中，外部环境对agent的奖励非常少或几乎不存在。在这种情况下，好奇心可以作为一种内在的奖励信号，让agent去探索未知的新环境，学习一些在以后生活中可能有用的技能。

## 外在动机和内在动机

外在动机和内在动机来源于心理学。

使用外部评价体系的人，对别人的评价特别在乎，甚至会内化别人对自己的评价，认为自己就是这样的。这样的人他们在做事情时，首先考虑的也是别人的看法。他们做事情的动力，常是为了博取别人的认可，这可以称为"外部动机"。

使用内部评价体系的人，对别人的评价不大在乎，他们做事情的动力来自于自己的内心，这可以称为"内部动机"。

## 利用内在动机促进探索

将内在奖励作为额外的探索奖励，即$r_{t}=r_{t}^{e}+\lambda r_{t}^{i}$，其中$\lambda$是调整探索与利用之间平衡的超参数。内在奖励就像agant的好奇心，促使agent去积极主动探索未知事物，以获得更多的知识和技能，这些知识和技能就可以作为以后学习的基础。

1. 基于计数的探索策略

   如果将状态的新颖程度作为内在奖励的条件，那就需要寻找一种方法来衡量当前状态是新颖的还是经常出现的。一种直观的方法是统计一个状态出现的次数，并相应地分配附加奖励。与经常出现的状态相比，附加奖励会引导智能体选择那些很少受访问的状态，这被称为基于计数的探索方法。

2. 基于预测的探索

   由预测模型预测agent行为的后果$f:(s_{t},a_{t})\to s_{t+1}$，其误差$e(s_{t},a_{t})=||f(s_{t},a_{t})-s_{t+1}||_{2}^{2}$作为内在探索奖励的标准，预测误差越高，我们对该状态的了解就越少，错误率下降得越快，我们获得的学习进度信号就越多。

# RL算法

> 参考链接：[强化学习 (Reinforcement Learning) | 莫烦Python (mofanpy.com)](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/)

## Q Learning

### 算法流程

<img src="https://static.mofanpy.com/results/reinforcement-learning/2-1-1.png" alt="2-1-1.png" style="background-color: white;" />

> Q表：表中每个值Q(s, a)表示在状态s下选择行为a的Q值。
>
> ϵ−greedy：值为0-1的探索概率。
>
> Off-policy：产生行为的策略和选取用于评估的行为的策略不一样的策略。
>
> On-policy：产生行为的策略和选取用于评估的行为的策略一致的策略。

1. 选择策略：每次有ϵ−greedy的概率选择Q(s,)中Q值最大的行为a（利用），也有(1-ϵ−greedy)的概率随机选择Q(s,)中的一个行为a（探索）；

2. 更新策略：假设此时状态为s，选择并执行完行为a，进入下一个状态s'，然后根据此时的s'，更新Q(s,a)的Q值；

3. 更新公式：

   $Q_{现实}=r+γmaxQ(s',)$

   $Q_{旧}=Q_{估计}=Q(s,a)$

   $Q(s,a)=Q_{旧}+α(Q_{现实}-Q_{估计})$

   其中，r为选择a所获得的奖励reward；γ为衰减值gamma；α为学习效率alpha；s'为选择完action后进入的下一个状态。

### 代码实现

极其简单的一个例子。

```python
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0
        S = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 环境更新
        while not is_terminated:

            A = choose_action(S, q_table)   # 选行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A]   # 估算的(状态-行为)值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table
```

## Sarsa

### 算法流程

仅与Q Learning在更新策略上存在区别。在Q Learning中，$Q_{现实}$ 中对状态s'预估的行为a'不一定会被执行，仅用于更新策略；在Sarsa中，$Q_{现实}$ 中对状态s'预估的行为a'就是下一步会被执行的行为，同时也用于更新策略。

![s4.png](https://static.mofanpy.com/results-small/ML-intro/s4.png)

1. 选择决策：每次有ϵ−greedy的概率选择Q(s,)中Q值最大的行为a（利用），也有(1-ϵ−greedy)的概率随机选择Q(s,)中的一个行为a（探索）；

2. 更新策略：假设此时状态为s，选择并执行完行为a，进入下一个状态s'，然后选择此时状态s'要做出的行为a'，并根据Q(s',a')的Q值更新Q(s,a)的Q值；

3. 更新公式：

   $Q_{现实}=r+γQ(s',a')$

   $Q_{旧}=Q_{估计}=Q(s,a)$

   $Q(s,a)=Q_{旧}+α(Q_{现实}-Q_{估计})$

### 代码实现

```python
def update():
    for episode in range(100):
        # 初始化环境
        observation = env.reset()

        # Sarsa 根据 state 观测选择行为
        action = RL.choose_action(str(observation))

        while True:
            # 刷新环境
            env.render()

            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)

            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            # 从 (s, a, r, s, a) 中学习, 更新 Q_tabel 的参数 ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个当成下一步的 state (observation) and action
            observation = observation_
            action = action_

            # 终止时跳出循环
            if done:
                break

    # 大循环完毕
    print('game over')
    env.destroy()
```

## Sarsa(lambda)

Sarsa(lambda)是在Sarsa的基础上进行的改进算法。Sarsa属于单步更新算法，每走一步更新上一个状态所选行为的Q值，而Sarsa(lambda)不再局限于单步更新，它每走一步就更新走到当前状态的所有已走过状态所选行为的Q值。

lambda的作用：lambda是一个值为0-1的衰减值，它表明离现在越远的状态的不可或缺性就越小，其对于获得最终目标的贡献就越小，因此其更新力度就应该越小。当lambda=0时就是单步更新，lambda=1时就是回合更新（不进行衰减，对于每一步的更新力度都一样）。当lambda在0和1之间，取值越大，离当前状态越近的步更新力度越大。

![sl5.png](https://static.mofanpy.com/results-small/ML-intro/sl5.png)

为了表明离当前状态越近的步更新力度越大，就需要为每步走过的地方加个标记值flag（比如1），这个flag会随着所走步数的增加而衰减（就是乘上lambda，即flag=flag×lambda），如果已经被标记的步又再走了一次，则会在原先标记值的基础上继续叠加一个flag（即flag=flag+1）。这里讲的衰减方法对应的就是下图中的第二个小图。

![3-3-2.png](https://static.mofanpy.com/results/reinforcement-learning/3-3-2.png)

如上图所示，第一个小图表示其中一个状态的action，它在一次episode中被经历的次数和时刻。第二个小图表示，这个行为经历一次就叠加一次flag，同时会伴随着衰减（直至0）。第三个小图是另一种衰减策略，每当这个行为被重新经历一次，它的flag不进行叠加，而只是重新回到最大值（比如1），然后同时也会伴随着衰减（直至0）。

### 算法流程

![3-3-1.png](https://static.mofanpy.com/results-small/reinforcement-learning/3-3-1.png)

1. 选择决策：每次有ϵ−greedy的概率选择Q(s,)中Q值最大的行为a（利用），也有(1-ϵ−greedy)的概率随机选择Q(s,)中的一个行为a（探索）；

2. 更新策略：假设此时状态为s，选择并执行完行为a，进入下一个状态s'，然后选择此时状态s'要做出的行为a'，并根据Q(s',a')的Q值更新所有已走过状态所选行为的Q值；

3. 更新公式：

   ```
   For all s∈S,a∈A(s):
   	Q(s,a)=Q(s,a)+α[(r+γQ(s',a'))-Q(s,a)]
   ```

### 代码实现

`eligibility_trace`是与Q表一样的表格，用于存储flag。

```python
class SarsaLambdaTable(RL): # 继承 RL class
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        ...
    def check_state_exist(self, state):
        ...
    def learn(self, s, a, r, s_, a_):
        # 这部分和 Sarsa 一样
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        # 第一种衰减策略：对于经历过的 state-action, 我们让他+1, 证明他是得到 reward 路途中不可或缺的一环
		self.eligibility_trace.ix[s, a] += 1

		# 第二种衰减策略（更有效的方式）
		# self.eligibility_trace.ix[s, :] *= 0
		# self.eligibility_trace.ix[s, a] = 1

        # Q table 更新
        self.q_table += self.lr * error * self.eligibility_trace

        # 随着时间衰减 eligibility trace 的值, 离获取 reward 越远的步, 他的"不可或缺性"越小
        self.eligibility_trace *= self.gamma*self.lambda_
```

## DQN

DQN是一种融合了神经网络和Q Learning的强化学习方法，完整名称叫做Deep Q Network。

在Q Learning中，我们使用表格来存储每一个状态state以及该状态的每个行为action所拥有的Q值，但当今问题太复杂，状态可以多到比天上的星星还多（比如下围棋）。如果全用表格来存储它们，恐怕我们的计算机有再大的内存都不够，而且每次在这么大的表格中搜索对应的状态也是一件很耗时的事。

一个有效的解决方法就是使用神经网络来代替表格，神经网络可以作为一种映射，即输入状态+行为（或仅有状态）输出对应的Q值，主要有两种方式：①  输入状态s+行为a，输出行为a的Q值Q(s,a)；② 仅输入状态s，输出该状态下所有行为的Q值Q(s,)。

![DQN2.png](https://static.mofanpy.com/results/ML-intro/DQN2.png)

经过神经网络分析后得到所有动作的Q值后，就可以直接按照Q Learning的原则，直接选择拥有最大值的动作当做下一步要做的动作。

![DQN4.png](https://static.mofanpy.com/results/ML-intro/DQN4.png)

为了切断相关性并加速神经网络的收敛，DQN还引入了两个利器：Experience replay（经验回放）和Fixed Q-targets（冻结Q现实）。

1. Experience replay是DQN的一种off-policy离线学习方法，它使得神经网络能学习当前经历着的，也能学习过去经历过的，甚至是学习别人的经历。因此，每次DQN更新的时候，我们都可以随机抽取一些之前的经历进行学习，而随机抽取的这种做法打乱了经历之间的相关性，使得神经网络的更新更有效率（在强化学习中，我们得到的观测数据是有序的，即step by step，用这样的数据去更新神经网络的参数会有问题）。具体做法就是创建一个记忆库Memory，存储经历过的数据，每次更新参数的时候从Memory中抽取一部分的数据来用于更新，以此打破数据间的关联。
2. Fixed Q-targets也是DQN的一种打乱经历之间相关性的方法，具体的方法就是将预测Q估计和Q现实的神经网络分成两个结构相同但参数不同的神经网络。一个是Q估计网络（eval_net），用于输出Q估计，该网络始终保持最新训练到的参数；另一个是Q现实网络（target_net），用于输出Q现实，该网络的参数是很久以前训练得到的，可以看成是Q估计网络的历史版本。

### 算法流程

首先环境给出一个observation，agent根据神经网络得到关于这个observation的所有Q(s,a)，然后利用ϵ−greedy策略选择action，环境接收到此action后给出一个奖励reward以及下一个observation_。这是一个step。此时我们根据reward去更新神经网络的参数，接着进入下一个step。如此循环下去，直到我们训练出了一个足够好的神经网络。

![img](https://wanjun0511.github.io/2017/11/05/DQN/dqn.png)

1. 如何更新Q估计网络的参数？

   eval_net的模型的输入为s，输出为Q(s,)。

   eval_net的损失函数为：

   ![img](https://pic4.zhimg.com/3858f07818d129668fc83d48d855bb1f_b.png)

   采用梯度下降的方式更新eval_net中每一层的参数w和b。

   更新用的（即损失函数中的）Q估计是s中选中的action的Q值，而Q现实是s_中Q值最大的action的Q值。

2. 如何更新Q现实网络的参数？

   每隔一定steps就将Q估计网络中的参数拷贝到Q现实网络中即可。

### 代码实现

```python
def run_maze():
    step = 0    # 用来控制什么时候学习
    for episode in range(300):
        # 初始化环境
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # DQN 根据观测值选择行为
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个 state, reward, 是否终止
            observation_, reward, done = env.step(action)

            # DQN 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 将下一个 state_ 变为 下次循环的 state
            observation = observation_

            # 如果终止, 就跳出循环
            if done:
                break
            step += 1   # 总步数

    # end of game
    print('game over')
    env.destroy()
```

其中进行神经网络训练`RL.learn()`的具体代码如下所示：

```python
def learn(self):
	# 检查是否替换 target_net 参数
	if self.learn_step_counter % self.replace_target_iter == 0:
    	self.sess.run(self.replace_target_op)
    	print('\ntarget_params_replaced\n')

	# 从 memory 中随机抽取 batch_size 这么多记忆
	if self.memory_counter > self.memory_size:
    	sample_index = np.random.choice(self.memory_size, size=self.batch_size)
	else:
    	sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
	batch_memory = self.memory[sample_index, :]

	# 获取 q_next (target_net 产生的 q) 和 q_eval(eval_net 产生的 q)
	q_next, q_eval = self.sess.run(
    	[self.q_next, self.q_eval],
    	feed_dict={
        	self.s_: batch_memory[:, -self.n_features:],
        	self.s: batch_memory[:, :self.n_features]
    	})

	# q_next, q_eval 包含了 s_ 和 s 中所有 action 的值，
	# 但我们只需要 s 中选中的 action 的值进行反向传递，
	# 所以只保留该 action 的值，而其他 action 值全变成0。
	# 此外，我们从 s_ 中选择的 action 与 s 中选择的 action 不一定在同一个位置上，
	# 因此，我们还需要调整 q_target，使 s_ 中选择的 action 的位置与 q_eval 中的一致。
	# 只有这样计算出来的 error 才是可以进行反向传递的。

	q_target = q_eval.copy()
	batch_index = np.arange(self.batch_size, dtype=np.int32)
	eval_act_index = batch_memory[:, self.n_features].astype(int)
	reward = batch_memory[:, self.n_features + 1]

	q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

	"""
	假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
    q_eval =
	[[1, 2, 3],
 	[4, 5, 6]]

	q_target = q_eval =
	[[1, 2, 3],
 	[4, 5, 6]]

	然后根据 memory 当中的具体 action 位置来修改 q_target 对应 action 上的值:
	比如在:
    	记忆 0 的 q_target 计算值是 -1, 而且我用了 action 0;
    	记忆 1 的 q_target 计算值是 -2, 而且我用了 action 2:
	q_target =
	[[-1, 2, 3],
 	[4, 5, -2]]

	所以 (q_target - q_eval) 就变成了:
	[[(-1)-(1), 0, 0],
 	[0, 0, (-2)-(6)]]

	最后我们将这个 (q_target - q_eval) 当成误差, 反向传递回神经网络。
	所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值，
	我们只反向传递之前选择的 action 的值。
	"""

	# 训练 eval_net
	_, self.cost = self.sess.run([self._train_op, self.loss],
           feed_dict={self.s: batch_memory[:, :self.n_features],self.q_target: q_target})
	self.cost_his.append(self.cost) # 记录 cost 误差

	# 逐渐增加 epsilon，降低行为的随机性
	self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
	self.learn_step_counter += 1
```

## Policy Gradients

强化学习有两种方法：一类是value-based方法，需要计算价值函数（value function），然后根据自己所认为的高价值来选择行为（action），比如Q Learning、Sarsa、DQN；另一类是policy-based方法，不需要计算value function，直接按照概率分布随机输出行为，比如PG。对比起以值为基础的方法，Policy Gradients直接输出行为的最大好处就是, 它能在一个连续区间内挑选动作，而基于值的方法一般都是只处理离散动作，无法处理连续动作。

这里讲PG的其中一个实现方法Monte-Carlo PG。该方法构造了一个策略神经网络（Policy Network），输入为observation，输出为该observation中所有行为的概率分布，然后就可以按照这个概率分布随机输出一个action（PG的策略就是概率大(帮助大或贡献多)的行为要多做）。该方法采用回合更新（也叫REINFORCE），即先存储一回合的记忆，然后根据这一回合的记忆去更新神经网络的参数。最后，通过更新神经网络参数，增加表现好的action出现的可能性，减小表现差的action出现的可能性。下面是PG的一个神经网络的示例：

![5-2-1.png](https://static.mofanpy.com/results/reinforcement-learning/5-2-1.png)

> PG基于以下假定：
>
> - 如果只在游戏终结时才有奖励和惩罚，该回合赢了，这个回合的所有样本都是"偏正的"，反之则该回合所有样本都是"偏负的"；
> - 距离赢的那刻越近贡献越大，越远贡献越小，一般采取指数衰减；
> - 如果在游戏的每个时刻都有奖励，那么当前行为的贡献就是前面每个时刻奖励的衰减累计之和。
>
> PG是按照概率分布来随机选择行为的，其中已经包含了探索部分。

PG如何进行反向传递？

1. PG是没有误差的，即PG不是采用loss函数来进行反向传递。实际上，PG用reward函数来代替loss函数，即PG追求的目标不是loss最小，而是reward最大，它通过reward函数来进行梯度下降，最终的目的就是让这次被选中的行为更有可能或更不可能在下次发生，也就是使用reward奖惩的方法来确定这个行为是不是应当增加被选的概率。

   ![PG03.png](https://static.mofanpy.com/results/ML-intro/PG03.png)

2. 举个例子。在上图中，我们通过神经网络分析，选出了左边的行为，我们直接进行反向传递，使之下次被选的可能性增加，但是奖惩信息却告诉我们，这次的行为是不好的，那这个动作可能性增加的幅度随之被减低。再比如这次的观测信息让神经网络选择了右边的行为，右边的行为随之想要进行反向传递，使右边的行为下次被多选一点，这时奖惩信息也来了，告诉我们这是好行为，那我们就在这次反向传递的时候加大力度，让它下次被多选的幅度更猛烈。

3. 现在的问题来到，如何构建reward函数来进行反向传递？reward函数推导过程略，我们直接看结果。在最终进行梯度下降时，神经网络参数θ的更新函数为$θ=θ+α∇_{θ}logπ_{θ}(s_{t},a_{t})v_{t}$，可以看出来它的梯度为delta(log(Policy(s,a))*v)，其表示在状态s对所选动作a的吃惊度。其中，Policy(s,a)为在状态s选择行为a的概率（值为0-1），log(Policy(s,a))表示Policy(s,a)这个事件的信息量，概率越小，信息量越大（从logx函数在0-1的递增性可以看出），则对应行为的概率的更新幅度就越大，反之亦然。此外，还要乘以一个与reward相关的参数v，它根据此次行为所获得的reward来表明该行为的被认可程度（吃惊程度），若该行为得到了一个大的reward，则v也大，说明该行为被大大的认可（或表示更吃惊），允许其进行大幅修改，反之则削弱其更新幅度。

4. PG的梯度下降也可以看成是将行为概率的更新幅度作为梯度来更新参数。

5. 如果将参数$v_{t}$的值在CartPole环境中一回合中每一步计算的值展现出来，则是下面这样的：

   <img src="https://static.mofanpy.com/results/reinforcement-learning/5-2-2.png" alt="5-2-2.png" style="zoom:50%;" />

   可以看出，左边一段的$v_{t}$有较高的值，右边的较低，这就是在说：请重视我这回合开始时的一系列动作，因为前面一段时间杆子还没有掉下来，而且请惩罚我之后的一系列动作，因为后面的动作让杆子掉下来了。这样$v_{t}$就能在每回合的学习中诱导gradient descent朝着正确的方向发展了。


### 算法流程

Policy Gradients算法的重点就是每一episode对神经网络参数θ进行一次梯度下降（θ=θ+α×梯度）。

> θ为神经网络中的参数。
>
> $\pi_{ \theta }$表示在θ神经网络中的策略函数Policy。
>
> α为learning rate。
>
> $∇_{θ}logπ_{θ}(s_{t},a_{t})v_{t}$为更新参数θ的梯度。

![5-1-1.png](https://static.mofanpy.com/results-small/reinforcement-learning/5-1-1.png)

### 代码实现

```python
for i_episode in range(3000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward) # 存储这一回合的 transition

        if done:
            ep_rs_sum = sum(RL.ep_rs) # 一回合下来总的reward

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True # 判断是否显示模拟
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn() # 学习，输出 vt

            if i_episode == 0:
                plt.plot(vt)    # plot 这个回合的 vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
```

RL.learn()的具体代码：

```python
def learn(self):
    # 衰减, 并标准化这回合的 reward
    discounted_ep_rs_norm = self._discount_and_norm_rewards()   # 功能再面

    # train on episode
    self.sess.run(self.train_op, feed_dict={
        self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
        self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
        self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
    })

    self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # 清空回合 data
    return discounted_ep_rs_norm    # 返回这一回合的 state-action value
```

选择action的具体代码：

```python
def choose_action(self, observation):
    # 所有 action 的概率
    prob_weights = self.sess.run(self.all_act_prob, feed_dict={
        self.tf_obs: observation[np.newaxis, :]})    
    # 根据概率来选 action
    action = np.random.choice(
        range(prob_weights.shape[1]), p=prob_weights.ravel())
    return action
```

梯度下降中参数$v_{t}$的计算方法：

```python
def _discount_and_norm_rewards(self):
    # discount episode rewards
    discounted_ep_rs = np.zeros_like(self.ep_rs)
    running_add = 0
    # 给每一个行为累积衰减的reward：
    # 一个行为的reward定义为从该reward开始到结束所获得的reward累积。
    # 此外，在累积过程需要乘以衰减因子gamma。
    for t in reversed(range(0, len(self.ep_rs))):
        running_add = running_add * self.gamma + self.ep_rs[t]
        discounted_ep_rs[t] = running_add

    # normalize episode rewards（正则化）
    discounted_ep_rs -= np.mean(discounted_ep_rs) # 减去平均值
    discounted_ep_rs /= np.std(discounted_ep_rs) # 除以方差
    return discounted_ep_rs
```

## Actor Critic

Actor Critic是在Policy Gradients算法的基础上改进的RL算法，由Actor和Critic组成。Actor就是一个和PG一样的策略神经网络，输入observation，输出该observation中所有行为的概率分布，然后就可以按照这个概率分布随机输出一个action作为当前状态的行为。

但与PG不一样的是，Actor在更新网络参数时对行为进行评价的参数（即PG中的$v_{t}$）变成了td_error，它的计算方法不再通过reward的衰减累积来求取，而是直接由Critic提供。Critic也是一个神经网络，输入state，就可以输出关于该state的每个行为的价值或者最优行为的价值（类似于Q Learning中的Q值），将该值与reward和下一个状态state_一起进行计算（与Q Learning中error的计算一样），得出td_error，然后将其作为Actor更新网络参数的评价值。

因此，Actor Critic对于PG的最大改进就是引入了Critic来学习环境和奖励之间的关系，使其能看到现在所处状态的潜在奖励，然后用它来指点Actor。此外，用Critic来指点Actor便能使Actor每一步都能进行更新，如果使用单纯的Policy Gradients，Actor就只能等到回合结束才能开始更新。

<img src="https://static.mofanpy.com/results/ML-intro/AC1.png" alt="AC1.png" style="zoom:67%;" /><img src="https://static.mofanpy.com/results/ML-intro/AC3.png" alt="AC3.png" style="zoom:67%;" />

可以看出，Actor的前身就是Policy Gradients，而Critic的前身就是Q Learning（或是其他Value-Based算法）。Actor可以毫不费力地在连续动作中选取合适的动作（Q Learning不行），Critic可以进行单步更新，从而使得Actor也能进行单步更新（单纯的Actor无法进行单步更新）。因此，两者的结合互相弥补了对方的缺陷。

但Actor Critic也有缺点，Actor Critic涉及到了两个神经网络，而且每次都是在连续状态中更新参数，每次参数更新前后都存在相关性，导致神经网络只能片面的看待问题，甚至导致神经网络学不到东西。Deep Deterministic Policy Gradient (DDPG)算法就是对Actor Critic的一种改进，即引入DQN中的经验回放和冻结来切断相关性。

总的来说，Actor Critic结合了Policy Gradient (Actor)和Function Approximation (Critic)的方法。Actor基于概率选行为，Critic基于Actor的行为评判行为的得分，Actor根据Critic的评分修改选行为的概率。通俗点就是，Actor修改行为时就像蒙着眼睛一直向前开车，Critic就是那个扶方向盘改变Actor开车方向的。或者说详细点就是，Actor在运用Policy Gradient的方法进行Gradient ascent的时候，由Critic来告诉他，这次的Gradient ascent是不是一次正确的ascent，如果这次的得分不好，那么就不要ascent那么多。

### 算法流程

总流程如下图所示：

<img src="https://static.mofanpy.com/results/reinforcement-learning/6-1-1.png" alt="6-1-1.png" style="zoom:80%;" />

Actor和Critic的神经网络如下所示：

<img src="https://static.mofanpy.com/results/reinforcement-learning/6-1-2.png" alt="6-1-2.png" style="zoom: 50%;" /><img src="https://static.mofanpy.com/results/reinforcement-learning/6-1-3.png" alt="6-1-3.png" style="zoom: 52%;" />

### 代码实现

Actor更新参数的代码：

```python
with tf.variable_scope('exp_v'):
    log_prob = tf.log(self.acts_prob[0, self.a])    # log 动作概率
    self.exp_v = tf.reduce_mean(log_prob * self.td_error)   # log 概率 * TD 方向
with tf.variable_scope('train'):
    # 因为我们想不断增加这个 exp_v (动作带来的额外价值),
    # 所以我们用过 minimize(-exp_v) 的方式达到 maximize(exp_v) 的目的
    self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)
```

Critic更新参数的代码：

```python
with tf.variable_scope('squared_TD_error'):
    self.td_error = self.r + GAMMA * self.v_ - self.v
    self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
with tf.variable_scope('train'):
    self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
```

每回合学习的代码：

```python
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []    # 每回合的所有奖励
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20    # 回合结束的惩罚

        track_r.append(r)

        # 每一步都进行 critic 和 actor 的学习
        td_error = critic.learn(s, r, s_)  # Critic 学习
        actor.learn(s, a, td_error)     # Actor 学习

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            # 回合结束，打印回合累积奖励
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
```

## DDPG

Deep Deterministic Policy Gradient (DDPG)吸收了Actor Critic让Policy gradient单步更新的精华，而且还吸收了DQN（使用一个记忆库和两套结构相同但参数更新频率不同的神经网络）切断相关性的精华。DDPG可以拆解成三部分来看，即Deep + Deterministic + PG。

1. Deep

   在神经网络的结构上和DQN一样，即Actor和Critic分别采用两套结构相同但参数更新频率不同的神经网络。对于Actor来说，就是actor_target_net和actor_eval_net；对于Critic，就是critic_target_net和critic_eval_net。此外，也采用和DQN一样的记忆库，用于存储足够多的记忆，然后在进行网络参数的更新时随机选取一部分进行学习。

   <img src="https://static.mofanpy.com/results-small/ML-intro/ddpg4.png" alt="ddpg4.png" style="zoom:67%;" />

   ① actor_eval_net：负责策略网络参数θ的迭代更新，并根据当前状态s选择当前动作a，用于和环境交互生成下一状态s_和奖励r。

   ② actor_target_net：负责根据经验回放池中采样的下一状态s\_选择最优下一动作a\_。网络参数θ'定期从θ复制。

   ③ critic_eval_net：负责价值网络参数w的迭代更新，并输出估计Q值Q(s,a,w)。

   ④ critic_target_net：负责计算现实Q值（$y_{i}=r+γQ′(s_,a_,w′)$）中的Q′(s\_,a\_,w')部分。网络参数w′定期从w复制。

2. Deterministic

   PG在选择行为时采用的是随机策略，而DDPG采用的是确定性策略。确定性策略的核心思想就是在同一个状态处，虽然各种行为的概率不同，但最大概率的只有一个，我们就只选取这个概率最大的行为就好了。

3. PG

   这个很好理解，DDPG就是在PG的基础上进一步改进得来的RL算法。

### 算法流程

<img src="https://static.mofanpy.com/results/reinforcement-learning/6-2-2.png" alt="6-2-2.png" style="zoom:67%;" /><img src="https://static.mofanpy.com/results/reinforcement-learning/6-2-3.png" alt="6-2-3.png" style="zoom:67%;" />

1. 每一回合将当前状态s输入actor_eval_net，得到行为a。执行行为a，得到新状态s\_和奖励r，将s、a、s\_和r保存到记忆库。令s=s\_，重复进行上述步骤，直到记忆库有足够多的样本；

2. 从记忆库随机抽取一些样本$\{s, a, s\_, r\}$进行学习：

   将s\_输入actor_target_net，得到s\_的最优行为a\_，然后将s\_和a\_输入critic_target_net，计算出现实Q值$y_{i}=r+γQ′(s_,a_,w′)$；

   将s和a输入critic_eval_net计算出估计Q值Q(s,a,w)；

   将得到的现实Q值和估计Q值一起计算出critic_eval_net的损失：

   $\frac{1}{m} \sum_{j=1}^{m}\left(y_{j}-Q\left(\phi\left(S_{j}\right), A_{j}, w\right)\right)^{2}$（均方差损失函数）

   然后通过神经网络的梯度反向传播来更新critic_eval_net的所有参数w。

   接着将估计Q值代入由确定性策略定义的梯度函数：

   $\left.\nabla J(\theta)=\left.\left.\frac{1}{m} \sum_{j=1}^{m}\left[\nabla_{a} Q\left(s_{i}\right., a_{i}, w\right)\right|_{s=s_{i}, a=\pi \theta(s)} \nabla_{\theta} \pi_{\theta(s)}\right|_{s=s_{i}}\right]$

   从而计算出actor_eval_net的损失，然后通过神经网络的梯度反向传播来更新actor_eval_net的所有参数θ。

3. 每隔一段时间，将actor_eval_net和critic_eval_net的参数更新到actor_target_net和critic_target_net中，这里的更新方法不是简单的复制，而是采取了软更新，即每次参数只更新一点点，即：

   $w^{\prime} \leftarrow \tau w+(1-\tau) w^{\prime}$

   $\theta^{\prime} \leftarrow \tau \theta+(1-\tau) \theta^{\prime}$

   其中$\tau$是更新系数，一般取的比较小，比如0.1或者0.01这样的值。

4. 不断循环上述过程，直到训练出足够好的Actor和Critic。

> 对梯度公式$\left.\nabla J(\theta)=\left.\left.\frac{1}{m} \sum_{j=1}^{m}\left[\nabla_{a} Q\left(s_{i}\right., a_{i}, w\right)\right|_{s=s_{i}, a=\pi \theta(s)} \nabla_{\theta} \pi_{\theta(s)}\right|_{s=s_{i}}\right]$的理解：
>
> 它的前半部分grad[Q]是来自于critic_eval_net，即Q估计，这是在说：这次Actor的行为要怎么移动，才能获得更大的Q。而后半部分grad[$\pi$] 是从actor_eval_net来的，这是在说：Actor 要怎么样修改自身参数，使得Actor更有可能做这个动作。所以，两者合起来就是在说：Actor要朝着更有可能获取更大Q的方向修改行为参数。

### 代码实现

DDPG类的代码实现：

```python
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
```

DDPG的训练代码：

```python
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        # 为了学习过程可以增加一些随机性，增加学习的覆盖，
        # DDPG 对选择出来的行为 a 会增加一定的噪声 var
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)
```



